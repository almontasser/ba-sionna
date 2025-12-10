# Configurable defaults (override via env or CLI args)
SCHEME=${SCHEME:-C3}
T=${T:-16}
EPOCHS=${EPOCHS:-12}
CDL_MODELS=${CDL_MODELS:-"A,C"}
TARGET_SNR=${TARGET_SNR:-5}

# Run training (additional args passed through)
python train.py \
  --scheme "$SCHEME" \
  -T "$T" \
  --epochs "$EPOCHS" \
  --cdl_models "$CDL_MODELS" \
  --target_snr "$TARGET_SNR" \
  "$@"

CDL_MODELS=${CDL_MODELS:-"A,B,C,D,E"}

# Run training (additional args passed through)
python train.py \
  --scheme "$SCHEME" \
  -T "$T" \
  --epochs "50" \
  --cdl_models "$CDL_MODELS" \
  --target_snr "20" \
  "$@"