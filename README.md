# Pipeline Code

Octo finetuning on a Franka FR3: GELLO teleop demos -> rosbags -> HDF5 -> TFDS -> finetuned checkpoint.

## Files

```
pipeline.sh / pipeline.env    pipeline driver + path/config settings (edit .env first)
extract_bags.py               rosbags → HDF5 episodes, 10 Hz grid
extract_bags_OldCam.py        same, for the old ~2.4 Hz camera bags (one step per camera
                              frame; action = command at the next frame)
build_dataset.py              HDF5 → TFDS (imports the external fr3_demo_dataset package)
fr3_dataset_builder.py        dataset builder
finetune_config.py            Octo finetuning config for the 10 Hz dataset
finetune_config_oldCam.py     same, for the ~2.4 Hz oldcam dataset (tfds_output_oldcam)
fr3_standardize_fn.py         maps dataset keys to Octo's expected obs/action format
Image_Publisher.py            camera-machine node: RealSense → ROS2 topics + MP4 recording
Franka_Bridge/                deployment: Octo checkpoint → FR3 joint impedance control
tfds_output_oldcam/           built TFDS dataset for the oldcam data (train 45 / val 6 eps)
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
Python 3.10 (`conda create -n octo_fr3 python=3.10`)

### Python packages

Install octo with its dependencies, then the CUDA build of JAX:


```bash
pip install -e <path-to-octo-repo>
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install protobuf==3.20.3 transformers==4.34.1 "setuptools<81"
pip install h5py opencv-python==4.8.1.78 pyyaml
python -c "import jax; print(jax.devices())"
```

- `transformers==4.34.1` — transformers 5.x dropped Flax, octo's T5 encoder crashes (`cannot import FlaxAutoModel`)
- `setuptools<81` — newer removes `pkg_resources`, which wandb 0.15.x needs

`octo` is installed editable. If the octo folder moves, re-run `pip install -e`.

### GPU

jaxlib 0.4.20 picks up whatever `ptxas` is first on PATH. If `ptxas --version` shows < CUDA 12 (or a GPU test fails with `CUDA_ERROR_INVALID_IMAGE` / "ptxas does not support CC ..."), put the pip-installed one first:

```bash
export PATH="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cuda_nvcc/bin:$PATH"
python -c "import jax.numpy as jnp; print((jnp.ones(3)+1).sum())"   # expect 6.0
```