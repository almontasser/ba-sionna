#!/usr/bin/env bash
# Minimal training wrapper: scenario, resume flag, and T only.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Defaults
SCENARIO="${SCENARIO:-UMi}"
T="${T:-16}"
RESUME="${RESUME:-1}"

# Allow scenario as first positional arg (UMi/UMa/RMa)
if [ "${1:-}" = "UMi" ] || [ "${1:-}" = "UMa" ] || [ "${1:-}" = "RMa" ]; then
  SCENARIO="$1"
  shift
fi

python train.py \
  --scenarios "${SCENARIO}" \
  --scenario_weights "${SCENARIO}=1" \
  --resume "${RESUME}" \
  -T "${T}"
