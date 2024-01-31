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

"""Training script."""

import functools
import gc
import os
import time
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from scipy.spatial.transform import Rotation
import gin
from inerf_helper import setup_model
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import camera_utils
from internal import utils
from internal import vis
from myvis import plot_cameras, plot_save_poses
import inerf_helper
import jax
from jax import jit, random
import jax.numpy as jnp
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.

def log_delta(path, c2w, clear=False) -> np.ndarray:
    Rot = c2w[..., :3, :3]
    T = c2w[..., :3, 3]
    euler_angles = [Rotation.from_matrix(R).as_euler('zyx', degrees=True) for R in Rot]

    path_fn = lambda x: os.path.join(path, x)
    
    mode = "w" if clear else "a"
    for i in range(T.shape[0]):
      fn = f"data_{i}.log"
      with open(path_fn(fn), mode) as f:
          euler_str = ",".join(map(str, euler_angles[i].ravel()))
          T_str = ",".join(map(str, T[i].ravel()))
          f.write(f"{euler_str},{T_str}\n")
    return euler_angles, T

def setup_model(
    config: configs.Config,
    rng: jnp.array,
    poseModel,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[models.Model, TrainState, Callable[
    [FrozenVariableDict, jnp.array, utils.Rays],
    MutableMapping[Text, Any]], Callable[
        [jnp.array, TrainState, utils.Batch, Optional[Tuple[Any, ...]], float],
        Tuple[TrainState, Dict[Text, Any], jnp.array]], Callable[[int], float]]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
  model, variables = models.construct_model(rng, dummy_rays, config)

  state, lr_fn = train_utils.create_optimizer(config, variables)
  render_eval_pfn = train_utils.create_render_fn(model)
  train_pstep = create_train_step(model, poseModel, config, dataset=dataset)

  return model, state, render_eval_pfn, train_pstep, lr_fn

def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  config = configs.load_config()

  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)

  np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
  cameras = tuple(np_to_jax(x) for x in dataset.cameras)
  

  if config.rawnerf_mode:
    postprocess_fn = test_dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z, _=None: z

  rng, key = random.split(rng)
  poseModel, poseState, pose_lr_fn = inerf_helper.setup_batch_model(config, rng, cameras[1].shape[0])

  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)
  poseState = checkpoints.restore_checkpoint(config.checkpoint_dir, poseState, prefix='poses_', step=250000)
  
  
  path_fn = lambda x: os.path.join(config.checkpoint_dir, x)

  log_dir = path_fn('data')
  if not utils.isdir(log_dir):
      utils.makedirs(log_dir)
  #log_delta(log_dir, cameras[1], clear=False)
  
  c2ws = poseModel.apply(poseState.params, cameras[1])
  
  #log_delta(log_dir, c2ws, clear=False)
  
  #plot_save_poses(cameras[1], c2ws)
  plot_cameras(cameras[1], c2ws)
  
if __name__ == '__main__':
  with gin.config_scope('train'):
    app.run(main)
