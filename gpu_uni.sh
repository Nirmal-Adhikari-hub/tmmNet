#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="your_wandb_api_key_here"

export CUDA_VISIBLE_DEVICES=0      # ← pin to the *other* GPU at the OS level
# --device "0" # <-- IMPORTANT: 0 means “the *first* visible GPU” (which is physical 1)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_RUN_NAME="resnet18_MotionDiffLearnHuge_tmm_v1_enable_motionTrue_preLstm_TMMalpha0"
export WANDB_RESUME=never

RUN=output/resnet18_MotionDiffLearnHuge_tmm_v1_enable_motionTrue_preLstm_TMMalpha0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

python main.py \
  --ablation_cfg configs/ablation_tmm_uni.yaml \
  --work-dir "$RUN" \
  2>&1 | tee "$RUN/train.log"