#!/bin/bash

run_experiment() {
  local delta_x=$1
  local delta_phi=$2
  local filter=$3
  local alpha0=$4
  local linear=$5

  python localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/home/yc/code/multinerf/datasets/real_data/'" \
    --gin_bindings "Config.checkpoint_dir = '/home/yc/code/multinerf/nerf_results/my_real_10/'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/yc/code/multinerf/nerf_results/localization_real/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.pose_lr_init = 0.01" \
    --gin_bindings "Config.pose_lr_final = 0.006" \
    --gin_bindings "Config.pose_sampling_strategy = 'random'" \
    --gin_bindings "Config.pose_delta_z = $delta_x" \
    --gin_bindings "Config.pose_delta_phi = $delta_phi" \
    --gin_bindings "Config.pose_w_alpha = $filter" \
    --gin_bindings "Config.pose_alpha0 = $alpha0" \
    --gin_bindings "Config.pose_alpha_linear = $linear" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count + 1))
}

# 初始化实验计数器
experiment_count=1

# 定义参数列表
delta_x_values=(0.5 1.0 2.0 4.0)
delta_phi_values=(2 4 8)
filter_values=("True" "False")

i=1
while [ $i -le 50 ]
do
  run_experiment 0.8 0 "True" 0.6 "False"
  i=$((i+1))
done

j=1
while [ $j -le 50 ]
do
  run_experiment 0.8 0 "False" 0.6 "False"
  j=$((j+1))
done

"""
for filter in "${filter_values[@]}"; do
  run_experiment $delta_x 0 $filter 0.6 "False"
done

# length=${#lr_init_values[@]}

for delta_x in "${delta_x_values[@]}"; do
  for filter in "${filter_values[@]}"; do
    run_experiment $delta_x 0 $filter 0.6 "False"
  done
done


for delta_phi in "${delta_phi_values[@]}"; do
  for filter in "${filter_values[@]}"; do
    run_experiment 0 $delta_phi $filter 0.6 "False"
  done
done

alpha0_values=(0.4 0.5 0.6 0.7 0.8)
for alpha0 in "${alpha0_values[@]}"; do
  for filter in "${filter_values[@]}"; do
    run_experiment "0.5" 0 "True" $alpha0 $filter
  done
done

i=50
while [ $i -le 5 ]
do
  run_experiment 2.0 0 $filter 0.6 "False"
  i=$((i+1))
done
"""