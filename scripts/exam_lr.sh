#!/bin/bash

run_experiment() {
  local method=$1

  python /home/air/multinerf/localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/data/yexy/From_external/multinerf/datasets/nerf1'" \
    --gin_bindings "Config.checkpoint_dir = '/data/yexy/From_external/multinerf/nerf_results/my_real_10/nerf1'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/air/multinerf/output/ablation/$experiment_count-alpha_linear($method)/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.batch_size = 1984" \
    --gin_bindings "Config.pose_max_steps = 1000" \
    --gin_bindings "Config.pose_lr_init = 0.01" \
    --gin_bindings "Config.pose_lr_final = 0.006" \
    --gin_bindings "Config.pose_sampling_strategy = 'random'" \
    --gin_bindings "Config.pose_delta_x = 0.5" \
    --gin_bindings "Config.pose_delta_y = 0.0" \
    --gin_bindings "Config.pose_delta_z = 0.0" \
    --gin_bindings "Config.pose_delta_phi = 0.0" \
    --gin_bindings "Config.pose_delta_theta = 0.0" \
    --gin_bindings "Config.pose_delta_psi = 0.0" \
    --gin_bindings "Config.pose_w_alpha = True" \
    --gin_bindings "Config.pose_optim_method = 'manifold" \
    --gin_bindings "Config.pose_alpha0 = 0.6" \
    --gin_bindings "Config.pose_alpha_linear = $method" \
    --gin_bindings "Config.pose_render_train = False" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count + 1))
}

# 初始化实验计数器
experiment_count=1
alpha_methods=(false true)

for i in {1..10}; do
  for method in "${alpha_methods[@]}"; do
    run_experiment $method
  done
done