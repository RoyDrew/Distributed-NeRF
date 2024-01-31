# Distributed NeRF


## Setup

```
# Clone the repo.
git clone https://github.com/RoyDrew/Distributed-NeRF.git
cd Distributed-NeRF

# Make a conda environment.
conda create --name multinerf python=3.9
conda activate multinerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Getting Started

To get started quickly with our pretrained model, you can use the provided script `scripts/simple_run.sh`. This script is designed to simplify the process of running the pretrained model with default configurations.

For more detailed information and additional results, navigate to the `output` directory.

## Running 

Example scripts for training, evaluating, and rendering can be found in
`scripts/`. You'll need to change the paths to point to wherever the datasets are located. 
For our model and some ablations can be found in `configs/`.

Summary: first, calculate poses. Second, train MultiNeRF. Third, render a result video from the trained NeRF model.

1. Calculating poses (using COLMAP):
```
DATA_DIR=my_dataset_dir
bash scripts/local_colmap_and_resize.sh ${DATA_DIR}
```
2. Training MultiNeRF:
```
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```
3. Rendering MultiNeRF:
```
python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr
```
Your output video should now exist in the directory `my_dataset_dir/render/`.

4. MultiNeRF Pose Optimization:
```
python -m localization \
  --gin_configs=configs/pose.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr
```

See below for more detailed instructions on either using COLMAP to calculate poses or writing your own dataset loader (if you already have pose data from another source, like SLAM or RealityCapture).

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

### Running COLMAP to get camera poses

In order to run MultiNeRF on your own captured images of a scene, you must first run [COLMAP](https://colmap.github.io/install.html) to calculate camera poses. You can do this using our provided script `scripts/local_colmap_and_resize.sh`. Just make a directory `my_dataset_dir/` and copy your input images into a folder `my_dataset_dir/images/`, then run:
```
bash scripts/local_colmap_and_resize.sh my_dataset_dir
```
This will run COLMAP and create 2x, 4x, and 8x downsampled versions of your images. These lower resolution images can be used in NeRF by setting, e.g., the `Config.factor = 4` gin flag.

By default, `local_colmap_and_resize.sh` uses the OPENCV camera model, which is a perspective pinhole camera with k1, k2 radial and t1, t2 tangential distortion coefficients. To switch to another COLMAP camera model, for example OPENCV_FISHEYE, you can run
```
bash scripts/local_colmap_and_resize.sh my_dataset_dir OPENCV_FISHEYE
```

If you have a very large capture of more than around 500 images, we recommend switching from the exhaustive matcher to the vocabulary tree matcher in COLMAP (see the script for a commented-out example).

Our script is simply a thin wrapper for COLMAP--if you have run COLMAP yourself, all you need to do to load your scene in NeRF is ensure it has the following format:
```
my_dataset_dir/images/    <--- all input images
my_dataset_dir/sparse/0/  <--- COLMAP sparse reconstruction files (cameras, images, points)
```


