#!/usr/bin/env bash
# pipeline.sh
# Written by Makhtar N.
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline: rosbags → HDF5 → TFDS → Octo finetuning
#
# Usage:
#   ./pipeline.sh all          # run all steps in order
#   ./pipeline.sh extract      # step 1 only: rosbags → HDF5
#   ./pipeline.sh build        # step 2 only: HDF5 → TFDS dataset
#   ./pipeline.sh finetune     # step 3 only: run Octo finetuning
#   ./pipeline.sh verify       # verify HDF5 files look correct
#   ./pipeline.sh clean        # remove intermediate HDF5 files (keep TFDS + checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail  # exit on error, undefined var, or pipe failure

# ── locate this script so relative paths work ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── load config ───────────────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/pipeline.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "✗ pipeline.env not found at $ENV_FILE"
    echo "  Copy and edit it before running this script."
    exit 1
fi
source "$ENV_FILE"

# ── colours ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"

log_step()  { echo -e "\n${GREEN}━━━ $1 ${NC}"; }
log_info()  { echo -e "${YELLOW}  → $1${NC}"; }
log_error() { echo -e "${RED}  ✗ $1${NC}"; }

# ── helper: require a variable is set and not a placeholder ───────────────────
check_path() {
    local var_name="$1"
    local value="$2"
    if [[ "$value" == "/path/to/"* ]]; then
        log_error "$var_name is still set to its placeholder value in pipeline.env"
        log_error "Edit pipeline.env before running."
        exit 1
    fi
}

validate_env() {
    check_path "BAG_DIR"        "$BAG_DIR"
    check_path "HDF5_DIR"       "$HDF5_DIR"
    check_path "TFDS_DIR"       "$TFDS_DIR"
    check_path "CHECKPOINT_DIR" "$CHECKPOINT_DIR"
    check_path "OCTO_DIR"       "$OCTO_DIR"

    if [[ ! -d "$BAG_DIR" ]]; then
        log_error "BAG_DIR does not exist: $BAG_DIR"
        exit 1
    fi
    if [[ ! -d "$OCTO_DIR" ]]; then
        log_error "OCTO_DIR does not exist: $OCTO_DIR"
        exit 1
    fi
}

# ── step 1: extract rosbags to HDF5 ──────────────────────────────────────────
step_extract() {
    log_step "Step 1 / 3 — Extracting rosbags → HDF5"

    mkdir -p "$HDF5_DIR"

    BAG_COUNT=$(find "$BAG_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
    log_info "Found $BAG_COUNT bag folders in $BAG_DIR"
    log_info "Writing HDF5 episodes to $HDF5_DIR"
    log_info "Language instruction: \"$LANGUAGE_INSTRUCTION\""

    python "${SCRIPT_DIR}/extract_bags.py" \
        --bag_dir              "$BAG_DIR" \
        --output_dir           "$HDF5_DIR" \
        --language_instruction "$LANGUAGE_INSTRUCTION"

    HDF5_COUNT=$(find "$HDF5_DIR" -name "*.hdf5" | wc -l)
    log_info "Extraction complete — $HDF5_COUNT HDF5 files written"
}

# ── step 1b: verify HDF5 files ────────────────────────────────────────────────
step_verify() {
    log_step "Verifying HDF5 files"

    HDF5_FILES=$(find "$HDF5_DIR" -name "*.hdf5" | sort)
    if [[ -z "$HDF5_FILES" ]]; then
        log_error "No HDF5 files found in $HDF5_DIR — run extract step first"
        exit 1
    fi

    # Use Python for the actual content check
    python - <<EOF
import h5py, os, glob, numpy as np

hdf5_dir = "$HDF5_DIR"
files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
print(f"  Checking {len(files)} HDF5 files...")

errors = 0
for fpath in files:
    try:
        with h5py.File(fpath, "r") as f:
            n_steps   = f["actions"].shape[0]
            act_dim   = f["actions"].shape[1]
            jp_shape  = f["observations/joint_positions"].shape
            img_shape = f["observations/images/primary"].shape
            lang      = f.attrs.get("language_instruction", "MISSING")

            ok = (act_dim == 8 and jp_shape[1] == 7 and img_shape[1:] == (256, 256, 3))
            status = "✓" if ok else "✗"
            print(f"  {status} {os.path.basename(fpath)}: "
                  f"{n_steps} steps | action={act_dim}D | joints={jp_shape[1]}D | "
                  f"img={img_shape[1:]} | lang='{lang}'")
            if not ok:
                errors += 1
    except Exception as e:
        print(f"  ✗ {os.path.basename(fpath)}: ERROR — {e}")
        errors += 1

print(f"\n  {'All files OK' if errors == 0 else f'{errors} file(s) with issues'}")
EOF
}

# ── step 2: build TFDS dataset ────────────────────────────────────────────────
step_build() {
    log_step "Step 2 / 3 — Building TFDS dataset"

    HDF5_COUNT=$(find "$HDF5_DIR" -name "*.hdf5" | wc -l)
    if [[ "$HDF5_COUNT" -eq 0 ]]; then
        log_error "No HDF5 files found in $HDF5_DIR — run extract step first"
        exit 1
    fi

    log_info "$HDF5_COUNT episodes → $TFDS_DIR"

    mkdir -p "$TFDS_DIR"

    # The builder needs to know where the HDF5 files are
    export HDF5_EPISODE_DIR="$HDF5_DIR"

    # fr3_dataset_builder.py must live inside a fr3_demo_dataset/ folder
    BUILDER_PARENT="${SCRIPT_DIR}"

    cd "$BUILDER_PARENT"
    python "${SCRIPT_DIR}/build_dataset.py" "$TFDS_DIR"

    log_info "TFDS build complete — data at $TFDS_DIR/fr3_demo_dataset/"
}

# ── step 3: run finetuning ────────────────────────────────────────────────────
step_finetune() {
    log_step "Step 3 / 3 — Finetuning Octo"

    # Confirm TFDS data exists
    TFDS_DATASET_DIR="${TFDS_DIR}/fr3_demo_dataset"
    if [[ ! -d "$TFDS_DATASET_DIR" ]]; then
        log_error "TFDS dataset not found at $TFDS_DATASET_DIR — run build step first"
        exit 1
    fi

    mkdir -p "$CHECKPOINT_DIR"

    log_info "Pretrained model : $PRETRAINED_PATH"
    log_info "Finetune mode    : $FINETUNE_MODE"
    log_info "Task modality    : $TASK_MODALITY"
    log_info "Saving to        : $CHECKPOINT_DIR"

    # Make the standardize fn importable
    export PYTHONPATH="${STANDARDIZE_FN_DIR}:${PYTHONPATH:-}"

    # Build wandb / debug args.
    # home/octo finetune.py uses --debug to set wandb mode="disabled".
    WANDB_ARGS=""
    if [[ "$WANDB_ENABLED" -eq 1 ]]; then
        WANDB_ARGS="--config.wandb.entity ${WANDB_ENTITY} --config.wandb.group ${WANDB_GROUP}"
    else
        WANDB_ARGS="--debug"
    fi

    cd "$OCTO_DIR"
    python scripts/finetune.py \
        --config "${SCRIPT_DIR}/finetune_config.py:${FINETUNE_MODE},${TASK_MODALITY}" \
        --config.pretrained_path "$PRETRAINED_PATH" \
        --config.save_dir        "$CHECKPOINT_DIR" \
        --config.dataset_kwargs.data_dir "$TFDS_DIR" \
        $WANDB_ARGS

    log_info "Finetuning complete — checkpoints at $CHECKPOINT_DIR"
}

# ── clean intermediate files ──────────────────────────────────────────────────
step_clean() {
    log_step "Cleaning intermediate HDF5 files"
    read -p "  Delete all .hdf5 files in $HDF5_DIR? [y/N] " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        rm -rf "$HDF5_DIR"
        log_info "Removed $HDF5_DIR"
    else
        log_info "Skipped."
    fi
}

# ── print summary before running ──────────────────────────────────────────────
print_summary() {
    echo -e "${GREEN}"
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │         Octo FR3 Finetuning Pipeline                │"
    echo "  └─────────────────────────────────────────────────────┘${NC}"
    echo "  Bags        : $BAG_DIR"
    echo "  HDF5        : $HDF5_DIR"
    echo "  TFDS        : $TFDS_DIR"
    echo "  Checkpoints : $CHECKPOINT_DIR"
    echo "  Mode        : ${FINETUNE_MODE} / ${TASK_MODALITY}"
    echo "  Instruction : \"$LANGUAGE_INSTRUCTION\""
    echo ""
}

# ── entrypoint ────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"

case "$COMMAND" in
    all)
        validate_env
        print_summary
        step_extract
        step_verify
        step_build
        step_finetune
        echo -e "\n${GREEN}  ✓ Pipeline complete!${NC}"
        ;;
    extract)
        validate_env
        step_extract
        ;;
    verify)
        validate_env
        step_verify
        ;;
    build)
        validate_env
        step_build
        ;;
    finetune)
        validate_env
        step_finetune
        ;;
    clean)
        validate_env
        step_clean
        ;;
    help|*)
        echo ""
        echo "Usage: ./pipeline.sh <command>"
        echo ""
        echo "Commands:"
        echo "  all       Run the full pipeline (extract → verify → build → finetune)"
        echo "  extract   Step 1: rosbags → HDF5 episode files"
        echo "  verify    Check HDF5 files for correct shapes and content"
        echo "  build     Step 2: HDF5 → TFDS/RLDS dataset"
        echo "  finetune  Step 3: run Octo finetuning"
        echo "  clean     Delete intermediate HDF5 files"
        echo ""
        echo "Configure paths and settings in pipeline.env before running."
        echo ""
        ;;
esac
