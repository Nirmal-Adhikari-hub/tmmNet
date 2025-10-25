#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1      # ← pin to the *other* GPU at the OS level
# --device "0" # <-- IMPORTANT: 0 means “the *first* visible GPU” (which is physical 1)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_RUN_NAME="resnet18_baseline"
export WANDB_RESUME=never

RUN=output/resnet18_baseline$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

python main.py \
  --work-dir "$RUN" \
  2>&1 | tee "$RUN/train.log"