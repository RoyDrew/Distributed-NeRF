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

import functools
import glob
import os
import time

from absl import app
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import camera_utils
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()

def load_models(config, rng):
    dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
    model, variables = models.construct_model(rng, dummy_rays, config)
    state, _ = train_utils.create_optimizer(config, variables)
    state1 = checkpoints.restore_checkpoint(config.checkpoint_dir1, state,step=150000)
    # 改一下config里面的path, 我这里就用重复的了
    state2 = checkpoints.restore_checkpoint(config.checkpoint_dir2, state, step=200000)
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
        render_dir = os.path.join(config.checkpoint_dir1, 'render')
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
    R = np.eye(3)
    t = np.ones((3,1))*0.01
    T = np.concatenate([R, t], axis=1)

    # 指定相机, TODO maybe? 将dataset分为 train dataset 和 test dataset
    cam_idx = 0

    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    # 设这里的data_dir 为 model1训练用的data_dir, 记为data_dir1
    data_dir1 = config.data_dir
    dataset1 = datasets.load_dataset('train', data_dir1, config)
    dataset1.set_single_cam_id(cam_idx)
    c2w1 = np_to_jax(dataset1.camtoworlds[cam_idx])

    data_dir2 = config.data_dir
    dataset2 = datasets.load_dataset('train', data_dir2, config)
    # 以下参数其实是一样的，可以不用
    p2c = np_to_jax(dataset2.pixtocams)
    # c2w2 = np_to_jax(dataset2.camtoworlds[cam_idx])
    # c2w2 应该由c2w1得到
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    c2w2 = T @ np.concatenate([c2w1, bottom], axis=0)

    distortion_params = np_to_jax(dataset2.distortion_params)
    p2c_ndc = np_to_jax(dataset2.pixtocam_ndc)
    camtype = dataset2.camtype

    rays1, pixels1 = dataset1.generate_ray_batch(cam_idx)
    rays1 = rays1.rays # 临时写, 抽了点
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
        
        chunk_renderings1, _ = render_fn1(chunk_rays1) # rng and alpha is deterministic
        chunk_renderings2, _ = render_fn2(chunk_rays2)

        # Unshard the renderings.
        chunk_renderings1 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), chunk_renderings1)
        chunk_renderings2 = jax.tree_util.tree_map(lambda v: utils.unshard(v[0], padding), chunk_renderings2)
        # 可以直接使用chunk_renderings里的ray_weights与ray_rgbs, 还有ray_sdist(不知道与delta是不是对应的?是距离吗)
        # ray_weights (render_chunk_size, n_samples)
        # ray_rgbs (render_chunk_size, n_samples, 3)
        # ray_sdist (render_chunk_size, n_samples+1)
        # 参照下面的代码将mipnerf360多次采样的weights等合并
        """
        keys = [k for k in rendering if k.startswith('ray_')]
        if keys:
            num_rays = rendering[keys[0]][0].shape[0]
            ray_idx = random.permutation(random.PRNGKey(0), num_rays)
            ray_idx = ray_idx[:config.vis_num_rays]
            for k in keys:
                rendering[k] = [r[ray_idx] for r in rendering[k]]
        """
        # 然后自己实现 merge_ray_samples https://github.com/ripl/nerfuser/blob/40093714557488f53c5e15534d46d92ab56403e2/nerfuser/view_blender.py#L172

        # merge
        chunk_rendering = {'rgb': 0.5 * chunk_renderings1[-1]['rgb'] + 0.5 * chunk_renderings2[-1]['rgb']}
        """
        for k in chunk_renderings1[0]:
            if k.startswith('ray_'):
                chunk_rendering1[k] = [r[k] for r in chunk_renderings1]

        for k in chunk_renderings2[0]:
            if k.startswith('ray_'):
                chunk_rendering2[k] = [r[k] for r in chunk_renderings2]
        """
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
