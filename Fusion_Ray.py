# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render script."""

from collections import defaultdict
import functools
import os
import time

from absl import app
from flax.training import checkpoints
from flax import linen as nn
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import camera_utils
from internal import utils
import jax
from jax import random, vmap
import jax.numpy as jnp
import numpy as np
from typing import Literal, Union
from functools import partial
from utils import get_noised_pose, extract_delta, find_Edge, find_EdgeRegion, create_alpha_fn
configs.define_common_flags()
jax.config.parse_flags_with_absl()

class WeightedRGBRenderer(nn.Module):
    """Weighted volumetic rendering."""
    background_color:  Union[Literal["random", "last_sample"], jax.Array] = "random"

    def __call__(self, rgb, ws, bg_w):
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            ws: Weighted termination probability mass for each sample.
            bg_w: Weighted termination probability mass for the background.

        Returns:
            Rendered RGB values.
        """
        comp_rgb = jnp.sum(rgb * ws, axis=-2)
        if self.background_color == 'last_sample':
            background_color = rgb[..., -1, :]
        elif self.background_color == 'random':
            background_color = jax.random.normal(comp_rgb.shape)
        else:
            background_color = self.background_color
        comp_rgb += background_color * bg_w
        return comp_rgb
    

def scatter(input, dim, index, src, reduce=None):
   # Works like PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
   
   dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
   
   if reduce is None:
       _scatter = jax.lax.scatter
   elif reduce == "add":
       _scatter = jax.lax.scatter_add
   elif reduce == "multiply":
       _scatter = jax.lax.scatter_mul
       
   _scatter = partial(_scatter, dimension_numbers=dnums)
   vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)
   vmap_outer = partial(vmap, in_axes=(1, 1, 1), out_axes=1)

   for idx in range(len(input.shape)):
       if idx == dim:
           pass
       elif idx < dim:
           _scatter = vmap_inner(_scatter)
       else:
           _scatter = vmap_outer(_scatter)
           
   return _scatter(input, jnp.expand_dims(index, axis=-1), src)

# https://github.com/ripl/nerfuser/blob/40093714557488f53c5e15534d46d92ab56403e2/nerfuser/view_blender.py#L172
def merge_ray_samples(weights, rgbs, deltas, scales):
    # weights (n_models, n_rays, n_samples, 1)
    # rgbs (n_models, n_rays, n_samples, 3)
    # deltas (n_models, n_rays, n_samples, 1)
    # scales (n_models)

    # merged_weights (n_models, n_rays, n_samples * n_models, 1)
    # merged_rgbs (n_models, n_rays, n_samples * n_models, 3)
    # merged_mids (n_rays, n_samples * n_models, 1)
    n_models, n_rays, n_samples = weights.shape[:3]
    merged_weights = jnp.zeros((n_models, n_rays, n_samples * n_models, 1))
    merged_rgbs = jnp.zeros((n_models, n_rays, n_samples * n_models, 3))
    merged_ends = jnp.zeros((n_rays, n_samples * n_models, 1))

    weights = jnp.concatenate((weights, jnp.zeros((n_models, n_rays, 1, 1))), axis=-2)
    rgbs = jnp.concatenate((rgbs, jnp.zeros((n_models, n_rays, 1, 3))), axis=-2)
    deltas = jnp.concatenate((deltas, jnp.full((n_models, n_rays, 1, 1), jnp.inf)), axis=-2)
    scales = scales[:, None, None, None]
    deltas *= scales
    ends = jnp.cumsum(deltas, axis=-2)
    ps = jnp.zeros((n_models, n_rays, 1, 1), dtype=int)
    for i in range(n_samples * n_models):
        end = jnp.take_along_axis(ends, ps, axis=-2)
        end_min = jnp.min(end, 0, keepdims=True)
        model_id = jnp.argmin(end, 0, keepdims=True)
        delta = end_min - (merged_ends[:, [i - 1]] if i else 0)
        merged_weights = merged_weights.at[:, :, [i]].set(delta / jnp.take_along_axis(deltas, ps, axis=-2) * jnp.take_along_axis(weights, ps, axis=-2))
        merged_rgbs = merged_rgbs.at[:, :, [i]].set(jnp.take_along_axis(rgbs, jnp.broadcast_to(ps, (n_models, n_rays, 1, 3)), axis=-2))
        merged_ends = merged_ends.at[:, [i]].set(end_min[0])
        ps = scatter(ps, 0, model_id, jnp.take_along_axis(ps, model_id, 0) + 1)
        
    merged_ends = jnp.concatenate((jnp.zeros((n_rays, 1, 1)), merged_ends), axis=-2)
    return merged_weights, merged_rgbs, (merged_ends[:, :-1] + merged_ends[:, 1:]) / 2 / scales

def idw(dists, g):
    t = jnp.nan_to_num(dists[:, None, ...] / dists, nan=1)
    return 1 / jnp.sum(t**g, axis=1)

def idw_with_filter(dists, g):
    threshold = 1.5
    t = jnp.nan_to_num(dists[:, None, ...] / dists, nan=1)
    tmp_result = 1 / jnp.sum(t**g, axis=1)
    
    t_12 = jnp.nan_to_num(dists[[0], ...] / dists[[1], ...], nan=1)
    
    mask_12 = t_12 > threshold #1远
    mask_12 = jnp.vstack([mask_12, mask_12])
    mask_21 = t_12 < 1/threshold #2远
    mask_21 = jnp.vstack([mask_21, mask_21])
    
    ones = jnp.ones((1, tmp_result.shape[1], tmp_result.shape[2]))
    zeros = jnp.zeros((1, tmp_result.shape[1], tmp_result.shape[2]))
    
    # use1 = jnp.concatenate((ones, zeros), axis=0)
    # use2 = jnp.concatenate((zeros, ones), axis=0)
    use2 = jnp.concatenate((ones, zeros), axis=0)
    use1 = jnp.concatenate((zeros, ones), axis=0)
    
    final_result = jnp.where(mask_12, use2, tmp_result)
    final_result = jnp.where(mask_21, use1, tmp_result)
    
    return final_result

def load_models(config, rng):
    dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
    model, variables = models.construct_model(rng, dummy_rays, config)
    state, _ = train_utils.create_optimizer(config, variables)
    state1 = checkpoints.restore_checkpoint(config.checkpoint_dir1, state)
    # 改一下config里面的path, 我这里就用重复的了
    state2 = checkpoints.restore_checkpoint(config.checkpoint_dir2, state)
    step = int(state.step)
    print(f'Load checkpoint at step {step}.')
    # 这里是为了避免每次重编译，加速渲染过程 | 好像没啥用...
    render_eval_pfn1 = train_utils.create_render_fn(model)
    render_eval_pfn2 = train_utils.create_render_fn(model)
    return model, state1, state2, render_eval_pfn1, render_eval_pfn2

def main(unused_argv):
    config = configs.load_config(save_config=False)
    render_dir = config.render_dir
    exam_id = config.pose_exam_id
    if render_dir is None:
        render_dir = os.path.join(config.checkpoint_dir, 'render')
    out_dir = os.path.join(render_dir, f'fusion_{exam_id}/')
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)
    path_fn = lambda x: os.path.join(out_dir, x)

    key = random.PRNGKey(20200823)
    # model一致，不一样的是modelState
    _, modelState1, modelState2, render_eval_pfn1, render_eval_pfn2 = load_models(config, key)

    render_fn1 = functools.partial(render_eval_pfn1, modelState1.params, 1.0, None, 1.0)
    render_fn2 = functools.partial(render_eval_pfn2, modelState2.params, 1.0, None, 1.0)


    # TODO load T_trans
    R = jnp.eye(3)
    t = jnp.ones((3,1))*0.01
    T = jnp.concatenate([R, t], axis=1)
#     T2=np.array([[-0.13140772, -0.9669068,   0.218685,    0.51096344],
#  [ 0.9911132,  -0.12354631,  0.04930461, -0.22784846],
#  [-0.02065524,  0.22322062,  0.974549,    0.00620916],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# final 7
#     T2=np.array([[-0.0966153 ,  0.85666686, -0.50673854,  0.6505079 ],
#  [-0.9953163 , -0.08481262 , 0.04638579 ,-0.25631362],
#  [-0.00324157 , 0.508847  ,  0.86084884 , 0.0763933 ],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # final 11
    # T2=np.array([[-9.7084627e-02,8.5688770e-01,-5.0627446e-01,6.5013921e-01],
    #     [-9.9527490e-01, -8.3566286e-02, 4.9411673e-02, -2.5620967e-01],
    #     [ 3.1264826e-05, 5.0868040e-01, 8.6095178e-01, 7.6511726e-02],
    #     [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # final 7 -STEP 1K
#     T2=np.array(  [[-0.1475093 , -0.971089  ,  0.1874015  , 0.795211  ],
#  [ 0.98905647 ,-0.1452624  , 0.02575502 ,-0.08219616],
#  [ 0.002226   , 0.18914887 , 0.98189163 ,-0.02151429],  
#         [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])   
  
    T2=np.array(    [[ 0.70025593, -0.7019439 ,  0.11805157 , 0.8760114 ],
 [ 0.70808333 , 0.6700287 , -0.21629049 , 0.28873044],
 [ 0.07265124,  0.23533  ,   0.96915066 ,-0.00971943],
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])   
        
    # 指定相机, TODO maybe? 将dataset分为 train dataset 和 test dataset
    cam_idx = 70

    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    # 设这里的data_dir 为 model1训练用的data_dir, 记为data_dir1
    data_dir1 = config.data_dir1
    dataset1 = datasets.load_dataset('test', data_dir1, config)
    #dataset1.set_single_cam_id(cam_idx)
    c2w1 = np_to_jax(dataset1.camtoworlds[cam_idx])
    delta = (config.pose_delta_x, config.pose_delta_y, config.pose_delta_z, config.pose_delta_phi, config.pose_delta_theta, config.pose_delta_psi)
    c2w1 = get_noised_pose(c2w1, delta)
    
    data_dir2 = config.data_dir1
    dataset2 = datasets.load_dataset('test', data_dir2, config)
    # 以下参数其实是一样的，可以不用
    p2c = np_to_jax(dataset2.pixtocams)
    # c2w2 = np_to_jax(dataset2.camtoworlds[cam_idx])
    # c2w2 应该由c2w1得到
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    c2w2 = T @ np.concatenate([c2w1, bottom], axis=0)
    c2w2 = T2
    

    distortion_params = np_to_jax(dataset2.distortion_params)
    p2c_ndc = np_to_jax(dataset2.pixtocam_ndc)
    camtype = dataset2.camtype

    _, pixels1 = dataset1.generate_ray_batch(cam_idx)
    #rays1 = rays1.rays
    _camera = (p2c, c2w1, distortion_params, p2c_ndc)
    rays1 = camera_utils.cast_ray_batch(_camera, pixels1, camtype, xnp=np)
    _camera = (p2c, c2w2, distortion_params, p2c_ndc)
    rays2 = camera_utils.cast_ray_batch(_camera, pixels1, camtype, xnp=np)

    # 以下为models.render_image 内容
    height, width = rays1.origins.shape[:2]
    num_rays = height * width
    rays1 = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays1)
    rays2 = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays2)

    host_id = jax.process_index()
    chunks = []
    idx0s = range(0, num_rays, config.render_chunk_size)
    for i_chunk, idx0 in enumerate(idx0s):
        # pylint: disable=cell-var-from-loop
        if i_chunk % max(1, len(idx0s) // 4) == 0:
            print(f'Rendering chunk {i_chunk}/{len(idx0s)-1}')
        chunk_rays1 = (jax.tree_util.tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], rays1))
        chunk_rays2 = (jax.tree_util.tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], rays2))

        actual_chunk_size = chunk_rays1.origins.shape[0]
        rays_remaining = actual_chunk_size % jax.device_count()
        if rays_remaining != 0:
            padding = jax.device_count() - rays_remaining
            chunk_rays1 = jax.tree_util.tree_map(lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays1)
            chunk_rays2 = jax.tree_util.tree_map(lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays2)
        else:
            padding = 0
        
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_rays1.origins.shape[0] // jax.process_count()
        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
        chunk_rays1 = jax.tree_util.tree_map(lambda r: utils.shard(r[start:stop]), chunk_rays1)
        chunk_rays2 = jax.tree_util.tree_map(lambda r: utils.shard(r[start:stop]), chunk_rays2)
        
        print("Rendering")
        chunk_renderings1, rays_history1 = render_fn1(chunk_rays1) # rng and alpha is deterministic
        _, rays_history2 = render_fn2(chunk_rays2)
        
        # 按batch_size对应同一条采样线
        rays_history1 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), rays_history1)
        rays_history2 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), rays_history2)
        # 可以直接使用rays_history2里的weights与rgb, 还有ray_tdist
        # ray_weights (render_chunk_size, n_samples)
        # ray_rgbs (render_chunk_size, n_samples, 3)
        # ray_sdist (render_chunk_size, n_samples+1)

        # Unshard the renderings. here is for images, RGBs are pixel rgb(e.g. after volume rendering)
        # chunk_renderings1 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), chunk_renderings1)
        # chunk_renderings2 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), chunk_renderings2)
        # 

        outputs = defaultdict(list)
        # (n_models, n_rays, n_samples, 1)
        outputs['weights'] = jnp.stack([rays_history1[-1]['weights'], rays_history2[-1]['weights']], axis=0)[..., None]
        # (n_models, n_rays, n_samples, 3)
        outputs['rgbs'] = jnp.stack([rays_history1[-1]['rgb'], rays_history2[-1]['rgb']], axis=0)
        # (n_models, n_rays, n_samples, 1) in metric ray distance
        outputs['deltas'] = jnp.stack([rays_history1[-1]['delta'], rays_history2[-1]['delta']], axis=0)[..., None]
        # (n_models, n_rays, 3)
        outputs['direction'] = jnp.stack([rays_history1[-1]['direction'], rays_history2[-1]['direction']], axis=0)

        #TODO scales
        scales = jnp.array([1, 1])
        # merged_weights (n_models, n_rays, n_samples * n_models, 1)
        # merged_rgbs (n_models, n_rays, n_samples * n_models, 3)
        # merged_mids (n_models, n_rays, n_samples * n_models, 1)
        merged_weights, merged_rgbs, merged_mids = merge_ray_samples(outputs['weights'], outputs['rgbs'], outputs['deltas'], scales)

        chunk_rendering = {}

        dummy_c2w_ts = jnp.concatenate([jnp.zeros((1,3)), t.reshape((1,3))], axis=0)

        # 下面是将这些权重全部使用IDW进行规范
        # poses orgin + directions * dist -> (n_models, n_rays, n_samples * n_models)
        # formulation (8)
        t_dirs = jnp.linalg.norm(outputs['direction'], axis=-1)[..., None, None]
        dists = jnp.linalg.norm(dummy_c2w_ts[:, None, None, :] + t_dirs * merged_mids, axis=-1)
        use_global_metric = False
        if use_global_metric:
            dists *= scales[:, None, None]
        # \gamma
        g = 5
        # IDW weighting (n_models, n_models, n_samples * n_models)
        # ws = idw(dists, g)
        ws = idw_with_filter(dists, g)
        # background weights (n_models, n_rays, 1)
        w_bg = ws[..., [-1]] # if bg == 'last_sample' else torch.full((n_keeps, n_rays, 1), 1 / n_keeps, device=ws.device)
        # IDW modified weights
        cs = merged_weights * ws[..., None] # (n_models, n_rays, n_models * samples, 1)
        # accumulation
        acc = outputs['weights'].sum(axis=-2) # (n_models, n_rays, 1)
        # The weight of the background.
        bg_w = jnp.maximum(0, 1 - acc)
        # IDW modified weight of the background.
        c_bg = bg_w * w_bg # (n_models, n_rays, 1)

        # s (1, n_rays, 1, 1) 用来归一化
        s = jnp.concatenate([cs, c_bg[..., None, :]], axis=-2).sum(axis=(0, -2), keepdims=True)
        c = 1 / (cs.shape[0] * (cs.shape[-2] + 1)) # n_models * (n_models * samples + 1)
        cs = jnp.nan_to_num(cs / s, nan=c) # (n_models, n_rays, n_models * samples, 1)
        c_bg = jnp.nan_to_num(c_bg / s.squeeze(-2), nan=c) # (n_models, n_rays, 1)

        comp_rgb = jnp.sum(merged_rgbs * cs, axis=-2)
        background_color = merged_rgbs[..., -1, :]
        comp_rgb += background_color * c_bg
        val = jnp.nan_to_num(jnp.clip(comp_rgb.sum(axis=0), 0, 1), nan=0) # (n_rays, 3)
        chunk_rendering['rgb'] = val
        
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = (jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *chunks))
    for k, z in rendering.items():
        if not k.startswith('ray_'):
        # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])
    utils.save_img_u8(rendering['rgb'], path_fn(f'0.png'))

    

    # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
    x = jax.numpy.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    print(x)


if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)
