#!/usr/bin/env bash
# If invoked as `sh train.sh`, re-exec under bash (arrays/pipefail are bash-only).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Train a single scenario model by default (UMi).
#
# Override knobs (optional):
#   RUN_NAME=final_curriculum EPOCHS=150 BATCH_SIZE=64 T=16 bash train.sh

SCENARIO="${SCENARIO:-UMi}"
RUN_NAME_ENV_SET=0
if [ -n "${RUN_NAME+x}" ]; then
  RUN_NAME_ENV_SET=1
fi
RUN_NAME="${RUN_NAME-}"
if [ -z "${RUN_NAME}" ]; then
  RUN_NAME="${SCENARIO}"
fi
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
T="${T:-16}"
RESUME="${RESUME:-1}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"

# Optional passthrough args to `train.py`:
#   bash train.sh --lr_schedule cosine_restarts --lr 2e-3 --cosine_first_decay_epochs 13
# Change scenario via argument:
#   bash train.sh UMa
if [ "${1:-}" = "UMi" ] || [ "${1:-}" = "UMa" ] || [ "${1:-}" = "RMa" ]; then
  SCENARIO="$1"
  if [ "${RUN_NAME_ENV_SET}" -eq 0 ]; then
    RUN_NAME="${SCENARIO}"
  fi
  shift
fi
EXTRA_ARGS=("$@")

COMMON=(
  --require_gpu
  --channel_gen_device gpu
  --train_channels_outside_graph 1
  --lr_schedule constant
  # --lr 0.002
  --lr 2e-3
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  -T "${T}"
  --resume "${RESUME}"
  --reset_optimizer "${RESET_OPTIMIZER}"
)

python train.py "${COMMON[@]}" \
  --run_name "${RUN_NAME}" \
  --scenarios "${SCENARIO}" \
  --scenario_weights "${SCENARIO}=1" \
  "${EXTRA_ARGS[@]}"
