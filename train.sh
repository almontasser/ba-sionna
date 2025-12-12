#!/bin/bash

# Fixed defaults (edit here to change)
T=16
EPOCHS=100
TARGET_SNR=20
LR_WARMUP_EPOCHS=2

# Warm-up settings
WARMUP_EPOCHS=3
WARMUP_CDL_MODELS="A,C"
WARMUP_SNR_RANGE="0,10"  # dB

# Main settings
MAIN_CDL_MODELS="A,B,C,D,E"
MAIN_SNR_RANGE=""  # empty = use Config defaults (-5,20)

# Checkpoint dir (shared across warm-up and main so training resumes)
CHECKPOINT_DIR=./checkpoints_C3_T${T}

echo "Warm-up: ${WARMUP_EPOCHS} epochs, CDL=${WARMUP_CDL_MODELS}, SNR=${WARMUP_SNR_RANGE}"
# python train.py \
#   -T "$T" \
#   --epochs "$WARMUP_EPOCHS" \
#   --cdl_models "$WARMUP_CDL_MODELS" \
#   --snr_train_range "$WARMUP_SNR_RANGE" \
#   --target_snr "$TARGET_SNR" \
#   --lr_warmup_epochs "$LR_WARMUP_EPOCHS" \
#   --checkpoint_dir "$CHECKPOINT_DIR" \
#   "$@"

echo "Main training: ${EPOCHS} epochs, CDL=${MAIN_CDL_MODELS:-default}, SNR=${MAIN_SNR_RANGE:-default}"
python train.py \
  -T "$T" \
  --epochs "$EPOCHS" \
  --cdl_models "${MAIN_CDL_MODELS}" \
  $( [ -n "$MAIN_SNR_RANGE" ] && printf -- '--snr_train_range %s ' "$MAIN_SNR_RANGE" ) \
  --target_snr "$TARGET_SNR" \
  --lr_warmup_epochs 0 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
