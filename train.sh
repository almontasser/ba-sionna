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
#   RUN_NAME=final_curriculum EPOCHS=150 BATCH_SIZE=64 T=16 bash train.sh

RUN_NAME="${RUN_NAME:-final_curriculum}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
T="${T:-16}"
SCENARIOS="${SCENARIOS:-UMi,UMa,RMa}"

RESUME="${RESUME:-1}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"

# Optional LR range test (set to 1 to run and exit).
LR_RANGE_TEST="${LR_RANGE_TEST:-0}"
LR_RANGE_MIN_LR="${LR_RANGE_MIN_LR:-1e-5}"
LR_RANGE_MAX_LR="${LR_RANGE_MAX_LR:-1e-2}"
LR_RANGE_STEPS="${LR_RANGE_STEPS:-2000}"
LR_RANGE_STOP_FACTOR="${LR_RANGE_STOP_FACTOR:-4.0}"

# Disable curriculum by setting: SCENARIO_CURRICULUM=0
SCENARIO_CURRICULUM="${SCENARIO_CURRICULUM:-1}"

# Default: balanced sampling across scenarios (used if curriculum is disabled).
SCENARIO_WEIGHTS="${SCENARIO_WEIGHTS:-UMi=0.333333,UMa=0.333333,RMa=0.333333}"

# Phase lengths (epochs). Main defaults to "EPOCHS - warmup1 - warmup2".
WARMUP1_EPOCHS="${WARMUP1_EPOCHS:-10}"
WARMUP2_EPOCHS="${WARMUP2_EPOCHS:-10}"
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
#   bash train.sh --lr_schedule cosine_restarts --lr 2e-3 --cosine_first_decay_epochs 13
# Special mode: pass "lr_test" or "lr_range_test" as the first arg to run the LR sweep.
if [ "${1:-}" = "lr_test" ] || [ "${1:-}" = "lr_range_test" ]; then
  LR_RANGE_TEST=1
  shift
fi
EXTRA_ARGS=("$@")

COMMON=(
  --require_gpu
  --channel_gen_device gpu
  --train_channels_outside_graph 1
  --lr_schedule cosine_restarts
  # --lr 0.003
  --lr 2e-3
  --cosine_first_decay_epochs 10
  --lr_warmup_epochs 3
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  -T "${T}"
  --scenarios "${SCENARIOS}"
  --run_name "${RUN_NAME}"
  --resume "${RESUME}"
  --reset_optimizer "${RESET_OPTIMIZER}"
)

if [ "${LR_RANGE_TEST}" = "1" ]; then
  # Use balanced scenario weights for the range test by default.
  python train.py "${COMMON[@]}" \
    --lr_range_test \
    --lr_range_min_lr "${LR_RANGE_MIN_LR}" \
    --lr_range_max_lr "${LR_RANGE_MAX_LR}" \
    --lr_range_steps "${LR_RANGE_STEPS}" \
    --lr_range_stop_factor "${LR_RANGE_STOP_FACTOR}" \
    --scenario_weights "${MAIN_WEIGHTS}" \
    "${EXTRA_ARGS[@]}"
  exit 0
fi

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
