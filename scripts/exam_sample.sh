#!/bin/bash

run_experiment() {
  local s=$1

  python localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/home/yc/code/multinerf/datasets/real_data/'" \
    --gin_bindings "Config.checkpoint_dir = '/home/yc/code/multinerf/nerf_results/my_real_10/'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/yc/code/multinerf/nerf_results/localization_real/sample/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.pose_lr_init = 0.002" \
    --gin_bindings "Config.pose_lr_final = 0.0006" \
    --gin_bindings "Config.pose_sampling_strategy = '$s'" \
    --gin_bindings "Config.pose_delta_z = 0.8" \
    --gin_bindings "Config.pose_delta_phi = 0" \
    --gin_bindings "Config.pose_w_alpha = True" \
    --gin_bindings "Config.pose_manifold = True" \
    --gin_bindings "Config.pose_alpha0 = 0.6" \
    --gin_bindings "Config.pose_alpha_linear = False" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count+1))
}

# 初始化实验计数器
experiment_count=3

s_values=("edge_region" "random" )

for s in "${s_values[@]}"; do
  run_experiment $s
done