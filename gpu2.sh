#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
RUN=output/resnet18_tmm_v2_preLstm_NoPhaseLoss_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"                          # ← create the folder before tee
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_RUN_NAME="resnet18_tmm_v2_preLstm_NoPhaseLoss"
# export WANDB_RESUME=never

python main.py \
  --ablation_cfg configs/ablation_tmm_gpu1.yaml \
  --work-dir "$RUN" \
  2>&1 | tee "$RUN/train.log"