import functools
import os
import flax
import gin
import datetime
from absl import app
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad
from flax.training import checkpoints
import matplotlib.pyplot as plt
from internal import camera_utils
from internal import train_utils
from internal import models
from internal import datasets
from internal import configs
from internal import utils
from inerf_helper import setup_model
from utils import get_noised_pose, extract_delta, find_Edge, find_EdgeRegion, create_alpha_fn

configs.define_common_flags()
jax.config.parse_flags_with_absl()

def create_train_step(model: models.Model,
                      modelState,
                      poseModel,
                      config: configs.Config,
                      p2c,
                      distortion_params,
                      p2c_ndc,
                      camtype,
                      c2w):
    
    def train_step(batch, poseState, alpha):
        def loss_fn(variables):
            rays = batch.rays
            _c2w = poseModel.apply(variables, c2w)
            _camera = (p2c, _c2w, distortion_params, p2c_ndc)
            rays = camera_utils.cast_ray_batch(_camera, rays, camtype, xnp=jnp)

            renderings, _ = model.apply(
                modelState.params,
                None,
                alpha,
                rays,
                train_frac=1.0,
                compute_extras=False,
                zero_glo=False)

            mse = jnp.mean(jnp.square(renderings[-1]['rgb'] - batch.rgb[..., :3]))
            return mse, _c2w
        
        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (mse, out_c2w), grad = loss_grad_fn(poseState.params)
        pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
        grad = pmean(grad)

        grad = train_utils.clip_gradients(grad, config)
        grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

        new_poseState = poseState.apply_gradients(grads=grad)

        return new_poseState, mse, out_c2w

    train_pstep = jax.pmap(
        train_step,
        axis_name='batch',
        in_axes=(0, 0, None))
    return train_pstep

def load_model(config, rng):
    dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
    model, variables = models.construct_model(rng, dummy_rays, config)
    state, _ = train_utils.create_optimizer(config, variables)
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    step = int(state.step)
    print(f'Load checkpoint at step {step}.')
    render_eval_pfn = train_utils.create_render_fn(model)
    return model, state, render_eval_pfn

def save_pose_config(out_dir, config):
    fn = os.path.join(out_dir, 'poseconfig.txt')
    with open(fn, 'w') as f:
        f.write(f'render_train={config.pose_render_train}\n')
        f.write(f'batch_size={config.batch_size}\n')
        f.write(f'patch_size={config.patch_size}\n')
        f.write(f'max_steps={config.pose_max_steps}\n')
        f.write(f'lr_init={config.pose_lr_init}\n')
        f.write(f'lr_final={config.pose_lr_final}\n')
        f.write(f'sampling_strategy={config.pose_sampling_strategy}\n')
        f.write(f'with_filter={config.pose_w_alpha}\n')
        f.write(f'optim_method={config.pose_optim_method}\n')
        f.write(f'alpha0={config.pose_alpha0}\n')
        f.write(f'delta_phi={config.pose_delta_phi}\n')
        f.write(f'delta_theta={config.pose_delta_theta}\n')
        f.write(f'delta_psi={config.pose_delta_psi}\n')
        f.write(f'delta_x={config.pose_delta_x}\n')
        f.write(f'delta_y={config.pose_delta_y}\n')
        f.write(f'delta_z={config.pose_delta_z}\n')

def cal_err(T1, T2):
    R1 = T1[:3,:3]
    t1 = T1[:3,3]
    R2 = T2[:3,:3]
    t2 = T2[:3,3]
    R21 = R1.T @ R2
    t21 = R1.T @ (t1 - t2)

    offset = np.linalg.norm(t21)

    angle = np.arccos((np.trace(R21) - 1) / 2)

    return offset, angle

def load_data(render_eval_pfn, state, config, cam_idx):
    dataset = datasets.load_dataset('test', config.data_dir, config)
    # data_property = dataset.load_data_property(cam_idx)

    obs_img, obs_img_c2w_1 = dataset.load_obs_data(cam_idx)
    return (obs_img, obs_img_c2w_1)

def main(unused_arg):
    config = configs.load_config(save_config=False)
    
    render_dir = config.render_dir
    exam_id = config.pose_exam_id
    out_dir = os.path.join(render_dir, f'{exam_id}/')
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)
    path_fn = lambda x: os.path.join(out_dir, x)
    
    save_pose_config(out_dir, config)
    
    rng = random.PRNGKey(0)
    model, modelState, render_eval_pfn = load_model(config, rng)

    # 指定相机
    cam_idx = 13
    
    obs_data = load_data(render_eval_pfn, modelState, config, cam_idx=0)
    # height, width, camera = data_property
    obs_img, obs_img_c2w_1 = obs_data

    dataset = datasets.load_dataset('train', config.data_dir, config)
    dataset.set_single_cam_id(cam_idx)
    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    p2c = np_to_jax(dataset.pixtocams)
    obs_img_c2w = np_to_jax(dataset.camtoworlds[cam_idx])
    distortion_params = np_to_jax(dataset.distortion_params)
    p2c_ndc = np_to_jax(dataset.pixtocam_ndc)
    camtype = dataset.camtype

    delta = (config.pose_delta_x, config.pose_delta_y, config.pose_delta_z, config.pose_delta_phi, config.pose_delta_theta, config.pose_delta_psi)
    start_c2w = get_noised_pose(obs_img_c2w, delta)

    poseModel, poseState, lr_fn = setup_model(config, rng, start_c2w)
    poseState = flax.jax_utils.replicate(poseState)
    
    POI = None

    train_vstep = create_train_step(model, modelState, poseModel, config, p2c, distortion_params, p2c_ndc, camtype, start_c2w)

    #是否在训练时render图片
    render_train = config.pose_render_train
    print("render in train:", render_train)
    def render(pose, alpha=1.0):
        rays = dataset.generate_ray_batch(cam_idx).rays
        _camera = (p2c, pose, distortion_params, p2c_ndc)
        rays = camera_utils.cast_ray_batch(_camera, rays, camtype, xnp=np)

        rendering = models.render_image(
            functools.partial(render_eval_pfn, modelState.params, 1.0),
            None, rays, config, alpha=alpha)
        rendered_img = rendering['rgb']
        return rendered_img
    
    if render_train:
            utils.save_img_u8(obs_img, path_fn(f'obs.png'))
    
    Nt = config.pose_max_steps
    #是否使用低通滤波器
    w_alpha = config.pose_w_alpha
    #初始频率阈值
    alpha0 = config.pose_alpha0

    #log输出文件
    fp = open(path_fn('train.log'), 'w')

    #创建alpha_fn
    alpha_fn = create_alpha_fn(alpha0, 0.95, config.pose_alpha_linear)

    #values to plot
    mse_values = []
    t_values = []
    angle_values = []

    pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
    # 迭代更新位姿
    for Nc, batch in zip(range(1, Nt+1), pdataset):

        alpha = alpha_fn(Nc/Nt) if w_alpha else 1.0
        lr = lr_fn(Nc)
        poseState, mse, out_c2w = train_vstep(batch, poseState, alpha)

        mse_values.append(mse[0])
        t, angle = cal_err(obs_img_c2w, out_c2w[0])
        t_values.append(t)
        angle_values.append(angle)

        # if Nc % 5 == 0:
        #     now = datetime.datetime.now()
        #     formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
        #     print(
        #         formatted_date, 
        #         f'{Nc}/{Nt}: '+
        #         f"mse={mse[0]:0.5f}, "+
        #         f"lr={lr:0.2e}, "+
        #         f"alpha={alpha:0.2e}, "+
        #         f"angle_error={angle:0.2e}, "+
        #         f"t_error={t:0.3e}", file=fp)
        #     fp.flush()
        if Nc % 10 == 0:
            now = datetime.datetime.now()
            formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
            print(
                formatted_date, 
                f'{Nc}/{Nt}: '+
                f"mse={mse[0]:0.5f}, "+
                f"angle_error={angle:0.2e}, "+
                f"t_error={t:0.3e},"+
                f"lr={lr:0.2e}, "+
                f"alpha={alpha:0.2e},"
                , file=fp)
            fp.flush()
        if Nc % 100 == 0 or Nc == 10:
            print(
                f"R,t={out_c2w[0]}\nref={obs_img_c2w}", file=fp)
            fp.flush()
            if render_train:
                rendered_img = render(out_c2w, alpha)
                utils.save_img_u8(rendered_img, path_fn(f'rendered_{Nc}.png'))

            plt.figure(1)
            plt.clf()
            plt.plot(mse_values)
            plt.ylabel('photo loss')
            plt.xlabel('iteration')
            plt.title('loss curve')
            plt.savefig(path_fn('p_loss.png'))
            
            plt.figure(2)
            plt.clf()
            plt.plot(t_values)
            plt.ylabel('transition offset')
            plt.xlabel('iteration')
            plt.title('loss curve')
            plt.savefig(path_fn('t_loss.png'))

            plt.figure(3)
            plt.clf()
            plt.plot(angle_values)
            plt.ylabel('rotation offset')
            plt.xlabel('iteration')
            plt.title('loss curve')
            plt.savefig(path_fn('angle_loss.png'))
        
    # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
    x = jax.numpy.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    print(x)

if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)