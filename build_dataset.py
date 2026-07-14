# Written by: Makhtar N.

import sys
import os

# Tell TFDS where our custom dataset lives
sys.path.insert(0, os.path.expanduser("~/octo_fr3"))

# Import the builder so TFDS registers it
import fr3_demo_dataset.fr3_dataset_builder  # noqa

import tensorflow_datasets as tfds

data_dir = sys.argv[1]

builder = tfds.builder("fr3_demo_dataset", data_dir=data_dir)
builder.download_and_prepare()
print("Dataset build complete.")
