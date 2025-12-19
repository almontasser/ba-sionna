# Deep Learning Based Adaptive Joint mmWave Beam Alignment

TensorFlow implementation of the paper "Deep Learning Based Adaptive Joint mmWave Beam Alignment" using **Sionna** for **3GPP TR 38.901** scenario channel modeling (**UMi/UMa/RMa**).

## Overview

This project implements a novel deep learning-based joint beam alignment scheme for millimeter wave (mmWave) communication systems that combines:
- **Codebook-based beam sweeping at the Base Station (BS)** using learned beam codebooks
- **Adaptive, codebook-free beam alignment at the User Equipment (UE)** using recurrent neural networks

The implementation automatically detects and uses the best available hardware: **CUDA GPU > Apple MPS > CPU**.

## Features

✅ End-to-end trainable beam alignment system  
✅ TR 38.901 scenario channel model (UMi/UMa/RMa) via Sionna  
✅ Explicit CIR → narrowband \(H\) mapping for paper measurement model  
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
cd /path/to/ba-sionna

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
├── channel_model.py         # Public channel-model wrapper (re-exports)
├── channels/                # TR 38.901 scenario channel implementation
├── metrics.py               # Performance metrics and baselines
├── baselines.py             # Baseline schemes (e.g., exhaustive search)
├── train.py                 # Training script
├── training/                # Training steps + LR schedules
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

# Per-scenario batch sampling (one scenario per batch; defaults to uniform weights)
python train.py --scenario_weights "UMi=0.7,UMa=0.2,RMa=0.1"
# (Equivalent explicit flags)
python train.py --w_umi 0.7 --w_uma 0.2 --w_rma 0.1

# Increase LR safely (multiplier) and label the run
python train.py --run_name lr_x3 --lr_scale 3.0

# Cosine warm restarts (useful if training plateaus after monotonic decay)
python train.py --run_name lr_cosine --lr_schedule cosine_restarts --lr 0.003 --cosine_first_decay_epochs 13

# Quick test run
python train.py --test_mode
```

See `QUICKSTART.md` and `VALIDATION_GUIDE.md` for recommended validation steps.

Train one robust multi-scenario model with 2 warmup stages + balanced main stage (default):
```bash
bash train.sh
```

Disable curriculum (pure balanced sampling for all epochs):
```bash
SCENARIO_CURRICULUM=0 RUN_NAME=final_balanced bash train.sh
```

Avoid accidental carry-over between runs:
- Start fresh: `RESUME=0 bash train.sh`
- Reuse weights but reset optimizer/LR state: `RESET_OPTIMIZER=1 bash train.sh`

Pass extra args through to `train.py`:
```bash
bash train.sh --lr_schedule cosine_restarts --lr 0.003 --cosine_first_decay_epochs 13
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

See `TENSORBOARD.md` for details (remote usage, tags, run directory layout).

### Evaluation

Generate paper-style plots (axes preserved):

```bash
# Generate both figures
python evaluate.py --figure all --num_samples 2000

# Figure 4: scenario comparison vs SNR
python evaluate.py --figure 4 --num_samples 2000

# Figure 5: scenario comparison vs T
python evaluate.py --figure 5 --num_samples 1000
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NTX` | 32 | Number of transmit antennas |
| `NRX` | 16 | Number of receive antennas |
| `T` | 16 | Number of sensing steps |
| `NCB` | 8 | BS codebook size |
| `BATCH_SIZE` | 256 | Training batch size (fits ~15 GB VRAM) |
| `EPOCHS` | 100 | Number of training epochs |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `RNN_TYPE` | "GRU" | UE RNN type (GRU/LSTM) |
| `RNN_NUM_LAYERS` | 2 | Number of recurrent layers |
| `RNN_HIDDEN_SIZE` | 256 | RNN hidden state size |
| `UE_BEAM_INDEX_ENCODING` | "one_hot" | How `x_t` is fed to the UE RNN |
| `UE_INCLUDE_TIME_FEATURE` | True | Add `t/(T-1)` as an RNN input |
| `MOBILITY_ENABLE` | True | Enable time-varying `H[t]` within an episode |

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
- Optimizer: Adam (default schedule: warm-up + exponential decay; optional cosine restarts)
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
- **Satisfaction Probability**: Pr[SNR_RX(dB) ≥ SNR_target(dB)] (paper Eq. 4–6)
- **Noise/SNR details**: see `NOISE_AND_SNR.md`

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

### Checkpoint Restore Errors (shape mismatch)
If you see an error like:
`Received incompatible tensor with shape ... when attempting to restore variable with shape ...`,
you are trying to load a checkpoint produced by a **different model definition** (e.g., different UE features/RNN size).

- Use a fresh directory: `python train.py --checkpoint_dir ./checkpoints_new_run`
- Or delete/rename the old checkpoint folder.
- See `VALIDATION_GUIDE.md` for details.

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
