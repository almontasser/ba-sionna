#!/usr/bin/env bash
set -euo pipefail

# LR + scenario-weight run script.
#
# Runs three training commands:
#  1) Baseline LR + UMi-heavy scenario sampling
#  2) LR ×3 + RMa-heavy scenario sampling
#  3) Cosine restarts + balanced scenario sampling (last)
#
# Override knobs (optional):
#   SEED=42 EPOCHS=50 BATCH_SIZE=128 T=16 SCENARIOS="UMi,UMa,RMa" bash train.sh

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
T="${T:-16}"
SCENARIOS="${SCENARIOS:-UMi,UMa,RMa}"

COMMON=(
  --require_gpu
  --channel_gen_device gpu
  --train_channels_outside_graph 1
  --val_fixed_channels 1
  --val_fixed_start_idx 1
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  -T "${T}"
  --scenarios "${SCENARIOS}"
)

# Step 1: UMi-heavy + baseline LR
python train.py --run_name "umi_heavy_lr_baseline" "${COMMON[@]}" \
  --scenario_weights "UMi=0.7,UMa=0.2,RMa=0.1" \
  --lr_schedule warmup_then_decay --lr 0.001 --lr_scale 1.0

# Step 2: RMa-heavy + LR ×3
python train.py --run_name "rma_heavy_lr_x3" "${COMMON[@]}" \
  --scenario_weights "UMi=0.1,UMa=0.2,RMa=0.7" \
  --lr_schedule warmup_then_decay --lr 0.001 --lr_scale 3.0

# Step 3 (last): balanced scenarios + cosine restarts
python train.py --run_name "balanced_lr_cosine_restarts" "${COMMON[@]}" \
  --scenario_weights "UMi=0.333333,UMa=0.333333,RMa=0.333333" \
  --lr_schedule cosine_restarts --lr 0.003 --lr_scale 1.0 \
  --cosine_first_decay_epochs 13 --cosine_t_mul 2.0 --cosine_m_mul 1.0 --cosine_alpha 0.0
