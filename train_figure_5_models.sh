#!/bin/bash
# Script to train C3 models for all T values needed for Figure 5

# T values from the paper for Figure 5
T_VALUES=(1 3 5 7 8 9 15)

echo "Training C3 models for Figure 5"
echo "T values: ${T_VALUES[@]}"
echo "==========================================="

echo ""
echo "Training C3 models..."
for T in "${T_VALUES[@]}"; do
    echo "  Training C3 with T=$T..."
    python train.py --num_sensing_steps $T --epochs 100
done

echo ""
echo "==========================================="
echo "Training complete!"
echo ""
echo "To generate Figure 5, run:"
echo "  python evaluate.py --figure 5 --num_samples 1000"
