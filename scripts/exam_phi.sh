#!/bin/bash

run_experiment() {
  local phi=$1
  local manifold=$2
  local tdlf=$3

  python localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/home/yc/code/multinerf/datasets/real_data/'" \
    --gin_bindings "Config.checkpoint_dir = '/home/yc/code/multinerf/nerf_results/my_real_10/'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/yc/code/multinerf/nerf_results/localization_real/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.pose_max_steps = 600" \
    --gin_bindings "Config.pose_lr_init = 0.01" \
    --gin_bindings "Config.pose_lr_final = 0.006" \
    --gin_bindings "Config.pose_sampling_strategy = 'random'" \
    --gin_bindings "Config.pose_delta_x = 0.0" \
    --gin_bindings "Config.pose_delta_y = 0.0" \
    --gin_bindings "Config.pose_delta_z = 0.0" \
    --gin_bindings "Config.pose_delta_phi = $phi" \
    --gin_bindings "Config.pose_delta_theta = 0" \
    --gin_bindings "Config.pose_delta_psi = 0" \
    --gin_bindings "Config.pose_w_alpha = $tdlf" \
    --gin_bindings "Config.pose_manifold = $manifold" \
    --gin_bindings "Config.pose_alpha0 = 0.6" \
    --gin_bindings "Config.pose_alpha_linear = False" \
    --gin_bindings "Config.pose_render_train = False" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count + 1))
}

# 初始化实验计数器
experiment_count=10

phi_values=(12)
manifold_values=("False" "True" "True")
tdlf_values=("False" "False" "True")

for phi in "${phi_values[@]}"; do
  for index in "${!manifold_values[@]}"; do
    run_experiment $phi ${manifold_values[$index]} ${tdlf_values[$index]} 
  done
done