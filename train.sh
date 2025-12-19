#!/usr/bin/env bash
# If invoked as `sh train.sh`, re-exec under bash (arrays/pipefail are bash-only).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Train one robust multi-scenario model by default with a 3-phase curriculum:
# warmup stage 1 -> warmup stage 2 -> main (balanced).
#
# Override knobs (optional):
#   RUN_NAME=final_balanced EPOCHS=100 BATCH_SIZE=128 T=16 bash train.sh

RUN_NAME="${RUN_NAME:-final_curriculum}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
T="${T:-16}"
SCENARIOS="${SCENARIOS:-UMi,UMa,RMa}"

RESUME="${RESUME:-1}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"

# Default: balanced sampling across scenarios (used if curriculum is disabled).
SCENARIO_WEIGHTS="${SCENARIO_WEIGHTS:-UMi=0.333333,UMa=0.333333,RMa=0.333333}"

# Curriculum schedule (phases in one run).
# Disable by setting: SCENARIO_CURRICULUM=0
SCENARIO_CURRICULUM="${SCENARIO_CURRICULUM:-1}"

# Phase lengths (epochs). Main defaults to "EPOCHS - warmup1 - warmup2".
WARMUP1_EPOCHS="${WARMUP1_EPOCHS:-3}"
WARMUP2_EPOCHS="${WARMUP2_EPOCHS:-3}"
MAIN_EPOCHS_DEFAULT=$((EPOCHS - WARMUP1_EPOCHS - WARMUP2_EPOCHS))
if [ "${MAIN_EPOCHS_DEFAULT}" -le 0 ]; then
  echo "ERROR: EPOCHS (${EPOCHS}) must be > WARMUP1_EPOCHS+WARMUP2_EPOCHS (${WARMUP1_EPOCHS}+${WARMUP2_EPOCHS})." >&2
  exit 1
fi
MAIN_EPOCHS="${MAIN_EPOCHS:-${MAIN_EPOCHS_DEFAULT}}"

# Phase weights (one scenario per batch; sampled across batches).
WARMUP1_WEIGHTS="${WARMUP1_WEIGHTS:-UMi=1,UMa=0,RMa=0}"
WARMUP2_WEIGHTS="${WARMUP2_WEIGHTS:-UMi=0.7,UMa=0.2,RMa=0.1}"
MAIN_WEIGHTS="${MAIN_WEIGHTS:-UMi=0.333333,UMa=0.333333,RMa=0.333333}"

CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-${WARMUP1_EPOCHS},${WARMUP2_EPOCHS},${MAIN_EPOCHS}}"
CURRICULUM_WEIGHTS="${CURRICULUM_WEIGHTS:-${WARMUP1_WEIGHTS};${WARMUP2_WEIGHTS};${MAIN_WEIGHTS}}"

# Optional passthrough args to `train.py`:
#   bash train.sh --lr_schedule cosine_restarts --lr 0.003 --cosine_first_decay_epochs 13
EXTRA_ARGS=("$@")

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
  --run_name "${RUN_NAME}"
  --resume "${RESUME}"
  --reset_optimizer "${RESET_OPTIMIZER}"
)

if [ "${SCENARIO_CURRICULUM}" = "1" ]; then
  python train.py "${COMMON[@]}" \
    --scenario_curriculum \
    --curriculum_epochs "${CURRICULUM_EPOCHS}" \
    --curriculum_weights "${CURRICULUM_WEIGHTS}" \
    "${EXTRA_ARGS[@]}"
else
  python train.py "${COMMON[@]}" \
    --scenario_weights "${SCENARIO_WEIGHTS}" \
    "${EXTRA_ARGS[@]}"
fi
