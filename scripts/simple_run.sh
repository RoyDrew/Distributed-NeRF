# python train.py \
#   --gin_configs=configs/real.gin \
#   --gin_bindings="Config.data_dir = 'dataset/real_45/nerf2'" \
#   --gin_bindings="Config.checkpoint_dir = 'ckpt/real_45'" \
#   --gin_bindings="Config.batch_size = 2048" \
#   --logtostderr

python render.py \
  --gin_configs=configs/real.gin \
  --gin_bindings="Config.data_dir = 'dataset/real_45/nerf2'" \
  --gin_bindings="Config.checkpoint_dir = 'ckpt/real_45'" \
  --gin_bindings="Config.render_path = False" \
  --gin_bindings="Config.render_path_frames = 30" \
  --gin_bindings="Config.render_dir = 'output/render_real'" \
  --gin_bindings="Config.render_video_fps = 10" \
  --logtostderr

python localization.py \
  --gin_configs=configs/pose.gin \
  --gin_bindings="Config.data_dir = 'dataset/real_45/nerf2'" \
  --gin_bindings="Config.checkpoint_dir = 'ckpt/real_45'" \
  --gin_bindings="Config.render_path = False" \
  --gin_bindings="Config.render_path_frames = 30" \
  --gin_bindings="Config.render_dir = 'output/localization_real'" \
  --gin_bindings="Config.render_video_fps = 10" \
  --logtostderr
