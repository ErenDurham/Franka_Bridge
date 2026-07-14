# octo_fr3 — pipeline code

Octo finetuning on a Franka FR3: GELLO teleop demos → rosbags → HDF5 → TFDS → finetuned checkpoint.

## Files

```
pipeline.sh / pipeline.env    pipeline driver + path/config settings (edit .env first)
extract_bags.py               rosbags → HDF5 episodes, 10 Hz grid
extract_bags_OldCam.py        same, for the old ~2.4 Hz camera bags (one step per camera
                              frame; action = command at the next frame)
build_dataset.py              HDF5 → TFDS (imports the external fr3_demo_dataset package)
fr3_dataset_builder.py        outdated draft builder — reference only, do not build with it
finetune_config.py            Octo finetuning config (modes: full | head_mlp_only | head_only)
fr3_standardize_fn.py         maps dataset keys to Octo's expected obs/action format
Image_Publisher.py            camera-machine node: RealSense → ROS2 topics + MP4 recording
Franka_Bridge/                deployment: Octo checkpoint → FR3 joint impedance control
Makefile                      convenience targets
```

## Usage

```bash
./pipeline.sh extract    # rosbags → HDF5
./pipeline.sh verify     # check HDF5 shapes/content
./pipeline.sh build      # HDF5 → TFDS
./pipeline.sh finetune   # finetune Octo
```

## Requirements

Python 3.10, then:

```bash
pip install numpy==1.24.3 ml-dtypes==0.2.0 scipy==1.11.4 protobuf==3.20.3 \
    tensorflow==2.15.0 tensorflow-datasets==4.9.2 tensorflow-metadata==1.13.0 \
    tensorflow-probability==0.23.0 flax==0.7.5 optax==0.1.5 chex==0.1.85 \
    distrax==0.1.5 "ml-collections>=0.1.0" h5py opencv-python==4.8.1.78 pyyaml \
    "wandb>=0.12.14" "transformers>=4.34.1"
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e <path-to-octo-repo> --no-deps
```