# Makefile — Octo FR3 finetuning pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Wraps pipeline.sh with short make targets.
#
# Usage:
#   make            # same as: make help
#   make all        # full pipeline
#   make extract    # step 1: rosbags → HDF5
#   make verify     # check HDF5 files
#   make build      # step 2: HDF5 → TFDS dataset
#   make finetune   # step 3: Octo finetuning
#   make clean      # remove HDF5 intermediates
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
SCRIPT := ./pipeline.sh

.PHONY: all extract verify build finetune clean help

help:
	@$(SCRIPT) help

all:
	@$(SCRIPT) all

extract:
	@$(SCRIPT) extract

verify:
	@$(SCRIPT) verify

build:
	@$(SCRIPT) build

finetune:
	@$(SCRIPT) finetune

clean:
	@$(SCRIPT) clean
