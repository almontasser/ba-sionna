# Deep Learning Based Adaptive Joint mmWave Beam Alignment

TensorFlow implementation of the paper "Deep Learning Based Adaptive Joint mmWave Beam Alignment" using Sionna for mmWave channel modeling.

## Overview

This project implements a novel deep learning-based joint beam alignment scheme for millimeter wave (mmWave) communication systems that combines:
- **Codebook-based beam sweeping at the Base Station (BS)** using learned beam codebooks
- **Adaptive, codebook-free beam alignment at the User Equipment (UE)** using recurrent neural networks

The implementation automatically detects and uses the best available hardware: **CUDA GPU > Apple MPS > CPU**.

## Features

✅ End-to-end trainable beam alignment system  
✅ Geometric mmWave channel model with Sionna support  
✅ Learned BS codebook with DFT initialization  
✅ Adaptive UE RNN controller (GRU/LSTM)  
✅ Automatic device selection (CUDA > MPS > CPU)  
✅ TensorBoard logging and visualization  
✅ Baseline comparisons (exhaustive search)  
✅ Reproduction of paper figures  

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+ (with GPU support if available)

### Setup

```bash
# Clone or navigate to the repository
cd /Users/almontasser/dev/beam-alignment

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check device availability
python device_setup.py

# Test configuration
python config.py
```

## Project Structure

```
beam-alignment/
├── config.py                 # System configuration and hyperparameters
├── device_setup.py          # Automatic device detection (CUDA/MPS/CPU)
├── utils.py                 # Utility functions (array response, beamforming)
├── channel_model.py         # mmWave geometric channel model
├── metrics.py               # Performance metrics and baselines
├── train.py                 # Training script
├── evaluate.py              # Evaluation and figure reproduction
├── models/
│   ├── bs_controller.py     # Base station with learned codebook
│   ├── ue_controller.py     # UE adaptive RNN controller
│   └── beam_alignment.py    # Complete end-to-end model
├── checkpoints/             # Saved model checkpoints
├── logs/                    # TensorBoard logs
└── results/                 # Evaluation results and plots
```

## Usage

### Training

Train the beam alignment model:

```bash
# Basic training
python train.py

# Training with custom parameters
python train.py --epochs 50 --batch_size 256 --lr 0.001

# Quick test run
python train.py --test_mode
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

### Evaluation

Reproduce paper results:

```bash
# Reproduce all figures
python evaluate.py --checkpoint_dir ./checkpoints --output_dir ./results

# Reproduce specific figure
python evaluate.py --mode fig6  # Figure 6: BF gain vs SNR
python evaluate.py --mode fig7  # Figure 7: BF gain vs sensing steps
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NTX` | 64 | Number of transmit antennas |
| `NRX` | 16 | Number of receive antennas |
| `T` | 8 | Number of sensing steps |
| `NCB` | 8 | BS codebook size |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 100 | Number of training epochs |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `RNN_TYPE` | "GRU" | UE RNN type (GRU/LSTM) |
| `RNN_HIDDEN_SIZE` | 128 | RNN hidden state size |

## Model Architecture

### Base Station Controller
- Learned beam codebook (trainable)
- DFT initialization for better convergence
- Sequential beam sweeping

### UE Controller
- GRU/LSTM for adaptive beam generation
- Inputs: received signal + BS beam index
- Outputs: combining vector + feedback

### Training
- Loss: Maximize beamforming gain
- Optimizer: Adam with exponential decay
- Gradient clipping for stability

## Device Support

The implementation automatically detects and uses the best available hardware:

1. **CUDA GPU** (NVIDIA): Fastest training
2. **MPS** (Apple Silicon): Good performance on M1/M2/M3 Macs
3. **CPU**: Fallback option

Check your device:

```bash
python -c "from device_setup import print_device_info; print_device_info()"
```

## Results

Expected performance (based on paper):
- **Beamforming Gain**: ~20 dB at SNR=5dB with T=8
- **vs Exhaustive Search**: Outperforms with fewer steps
- **Satisfaction Probability**: Increases with more sensing steps

## Testing Components

Test individual components:

```bash
# Test channel model
python channel_model.py

# Test BS controller
python models/bs_controller.py

# Test UE controller
python models/ue_controller.py

# Test complete model
python models/beam_alignment.py

# Test metrics
python metrics.py

# Test utilities
python utils.py
```

## Paper Reference

```bibtex
@article{tandler2024beam,
  title={Deep Learning Based Adaptive Joint mmWave Beam Alignment},
  author={Tandler, Daniel and Gauger, Marc and Tan, Ahmet Serdar and Dörner, Sebastian and ten Brink, Stephan},
  journal={arXiv preprint arXiv:2401.13587},
  year={2024}
}
```

## License

This implementation is for research and educational purposes.

## Troubleshooting

### Out of Memory Errors
- Reduce `BATCH_SIZE` in `config.py`
- Enable memory growth for GPU in `device_setup.py`

### Slow Training on CPU
- Consider using cloud GPU (Google Colab, AWS, etc.)
- Reduce model size or dataset

### Sionna Import Errors
```bash
pip install --upgrade sionna
```

### TensorFlow GPU Not Detected
```bash
# For NVIDIA GPU
pip install tensorflow[and-cuda]

# For Apple Silicon
# TensorFlow 2.10+ includes MPS support by default
```

## Contributing

Contributions are welcome! Areas for improvement:
- Full Sionna channel model integration
- Additional baseline implementations
- Hyperparameter tuning
- Multi-GPU training support

## Contact

For questions or issues, please open an issue on the repository.

---

**Happy Beam Alignment! 📡**
