"""
Configuration File for mmWave Beam Alignment System

This module defines all hyperparameters, system settings, and experimental
configurations for the beam alignment system. All parameters are based on
the specifications from "Deep Learning Based Adaptive Joint mmWave Beam
Alignment" (arXiv:2401.13587v1).

Configuration Categories:

1. Antenna Array Parameters:
   - NTX: Number of BS transmit antennas (32 per paper)
   - NRX: Number of UE receive antennas (16)
   - Antenna spacing: Half-wavelength (λ/2)

2. Channel Model Parameters (Sionna TR 38.901 scenarios):
   - CARRIER_FREQUENCY: 28 GHz (mmWave band)
   - 3GPP TR 38.901 stochastic scenarios (UMi/UMa/RMa)

3. Beam Alignment Parameters:
   - T: Number of sensing steps (16 in paper figures)
   - NCB: BS codebook size (8 beams)
   - Start beam selection: fixed or random

4. Training Parameters:
   - BATCH_SIZE: 256 (set for 15 GB VRAM)
   - EPOCHS: 100
   - LEARNING_RATE: 0.001 with exponential decay
   - SNR_TRAIN: 10 dB training SNR
   - SNR_TEST_RANGE: -10 to 20 dB for evaluation

5. Neural Network Architecture:
   - RNN_TYPE: "GRU" (paper default)
   - RNN_NUM_LAYERS: 2 (paper default)
   - RNN_HIDDEN_SIZE: 256 (default in this repo; configurable)
   - NUM_FEEDBACK: 16 feedback values

Usage:
    As a class with static attributes:
        >>> from config import Config
        >>> print(Config.NTX, Config.NRX)
        32 16
        >>> Config.print_config()  # Print all settings

    Modify for specific runs:
        >>> Config.BATCH_SIZE = 512
        >>> Config.EPOCHS = 200

Note:
    The configuration is designed to match the paper's experimental setup.
    Changing these values may affect reproducibility of paper results.
    
References:
    Paper Section IV: Simulation Setup
    Paper Table I: System Parameters
"""

import numpy as np


class Config:
    """Configuration parameters for the beam alignment system"""
    
    # ==================== Antenna Array Parameters ====================
    NTX = 32  # Number of transmit antennas at BS (per arXiv paper)
    NRX = 16  # Number of receive antennas at UE
    
    # ==================== Channel Model Parameters (Sionna TR 38.901 scenarios) ====================
    CARRIER_FREQUENCY = 28e9  # 28 GHz (mmWave)
    WAVELENGTH = 3e8 / CARRIER_FREQUENCY  # Speed of light / frequency
    ANTENNA_SPACING = WAVELENGTH / 2  # Half-wavelength spacing for ULA
    
    # Domain Randomization Parameters (for robust training)
    # Scenarios: Urban Micro (UMi), Urban Macro (UMa), Rural Macro (RMa)
    SCENARIOS = ["UMi", "UMa", "RMa"]
    # O2I model for UMi/UMa (required by Sionna, even if indoor_probability=0)
    O2I_MODEL = "low"  # {"low","high"}
    # If enabled, SNR_RX depends on distance/pathloss; keep disabled for paper-style normalization.
    ENABLE_PATHLOSS = False
    ENABLE_SHADOW_FADING = False

    # Topology sampling (1 UT + 1 BS per sample)
    # NOTE: With pathloss disabled, this mostly affects LOS probability and cluster statistics.
    DISTANCE_RANGE_M = (10.0, 200.0)  # UT-BS 2D distance range
    UE_SPEED_RANGE = (0.0, 30.0)  # 0 to 30 m/s (0 to 108 km/h)
    INDOOR_PROBABILITY = 0.0  # probability the UT is indoor (affects O2I for UMi/UMa)
    UT_HEIGHT_M = 1.5
    BS_HEIGHT_UMI_M = 10.0
    BS_HEIGHT_UMA_M = 25.0
    BS_HEIGHT_RMA_M = 35.0

    # OFDM resource grid for Sionna channel generation
    RESOURCE_GRID_FFT_SIZE = 64
    RESOURCE_GRID_SUBCARRIER_SPACING = 120e3  # Hz

    # ==================== Mobility / Time Variation ====================
    # If enabled, the channel model generates a *time-varying* narrowband channel
    # sequence H[t] (t=0..T), and the sensing measurements use H[t] at step t.
    #
    # This goes beyond the common quasi-static-within-an-episode assumption by
    # letting the UE move and inducing Doppler/time evolution during the T sensing
    # steps.
    #
    # References (URLs added per request):
    # - TR 38.901 channel model (official 3GPP spec archive):
    #   https://www.3gpp.org/ftp/specs/archive/38_series/38.901
    # - Jaeckel et al., spatial consistency for the 3GPP NR channel model:
    #   https://arxiv.org/abs/1808.04659
    # - Hoydis et al., “Sionna: An Open-Source Library for Next-Generation Physical Layer Research”:
    #   https://arxiv.org/abs/2203.11854
    MOBILITY_ENABLE = True
    # Number of TR 38.901 time samples per episode. If None, uses (T+1) so that
    # we have one channel snapshot for each sensing step and one for the final
    # beamforming step.
    MOBILITY_NUM_TIME_SAMPLES = None
    # Sampling frequency [Hz] for time evolution, i.e., t_n = n / sampling_frequency.
    # A reasonable default is to align the time step with one OFDM symbol duration
    # (ignoring CP): dt ≈ 1/subcarrier_spacing.
    MOBILITY_SAMPLING_FREQUENCY_HZ = RESOURCE_GRID_SUBCARRIER_SPACING

    # ==================== Performance / Device Placement ====================
    # TR 38.901 scenario channel generation can be expensive and can run on CPU or GPU.
    # Note: The channel generator uses Python control flow (via tf.py_function in graph
    # mode), so it will never be "pure GPU" end-to-end, but its TensorFlow ops can
    # still execute on GPU when this is set to "gpu"/"auto".
    CHANNEL_GENERATION_DEVICE = "auto"  # {"auto","cpu","gpu"}

    # If True, training generates channels *outside* the graph-compiled train step.
    # This avoids `tf.py_function` inside `@tf.function`, which would otherwise force
    # the channel tensor to be produced on CPU and then copied to GPU.
    TRAIN_CHANNELS_OUTSIDE_GRAPH = True
    
    # ==================== Beam Alignment Parameters ====================
    T = 16  # Number of sensing steps (Paper uses T=16 for Figure 4)
    NCB = 8  # Codebook size at BS (number of beams in learned codebook)
    # C3 default: random sweep start index i ~ Uniform{0..NCB-1}
    RANDOM_START = True
    START_BEAM_INDEX = 0  # Used only when RANDOM_START=False
    
    # ==================== Training Parameters ====================
    BATCH_SIZE = 256  # Reduced to fit ~15 GB VRAM comfortably
    EPOCHS = 100
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.96
    LEARNING_RATE_DECAY_STEPS = 10  # Decay every 10 epochs
    LR_WARMUP_EPOCHS = 0  # Linear warm-up epochs (0 disables warm-up)

    # Loss configuration
    # "paper": maximize normalized linear gain (Eq. 7 in paper)
    # "log": optional surrogate for ablations/stability
    LOSS_TYPE = "paper"

    # ==================== Narrowband Mapping ====================
    # How to reduce the TR 38.901 CIR/frequency response to a narrowband H used in y_t.
    # "center": use DC/center-subcarrier (paper-consistent flat fading)
    # "subcarrier": pick a specific subcarrier index
    # "mean_cfr": average CFR over all subcarriers (kept for ablations)
    NARROWBAND_METHOD = "center"
    NARROWBAND_SUBCARRIER = None  # int index if method=="subcarrier"
    
    # SNR parameters
    SNR_TRAIN = 10.0  # Training SNR in dB (per arXiv paper)
    SNR_TEST_RANGE = np.arange(-10, 21, 2)  # Test SNR range for evaluation
    SNR_TARGET = 20.0  # Target SNR for satisfaction probability (dB) (per arXiv paper)
    
    # Domain Randomization for SNR (set SNR_TRAIN_RANGE to enable)
    SNR_TRAIN_RANGE = (-5.0, 20.0)  # Random SNR range for robust training (dB)
    SNR_TRAIN_RANDOMIZE = True  # Enable SNR randomization during training
    
    # UE Controller (RNN) parameters
    RNN_TYPE = "GRU"  # Paper uses GRU (2-layer Gated Recurrent Units)
    RNN_NUM_LAYERS = 2  # Paper: two recurrent layers
    # NOTE: 384 hidden units makes the UE RNN >1M parameters. 256 is closer to
    # the paper-scale model size while remaining expressive under multi-scenario
    # TR 38.901 + mobility.
    RNN_HIDDEN_SIZE = 256
    RNN_DROPOUT = 0.0
    RNN_RECURRENT_DROPOUT = 0.0
    NUM_FEEDBACK = 16  # Number of feedback values (NFB) (per arXiv paper)

    # UE input feature engineering
    # "scalar": x_t normalized to [0,1]
    # "one_hot": one-hot encoding of x_t (often improves learning stability)
    UE_BEAM_INDEX_ENCODING = "one_hot"
    UE_INCLUDE_TIME_FEATURE = True  # include t/T as an extra input feature
    UE_INPUT_LAYER_NORM = True
    UE_OUTPUT_LAYER_NORM = False
    UE_DROPOUT_RATE = 0.0

    # BS feedback network (N2) sizing
    BS_FNN_HIDDEN_SIZES = (256, 256)
    BS_FNN_ACTIVATION = "gelu"
    BS_FNN_LAYER_NORM = True
    
    # ==================== Data Generation ====================
    NUM_TRAIN_SAMPLES = 100000  # Increased from 50K for better training
    NUM_VAL_SAMPLES = 10000  # Increased proportionally
    NUM_TEST_SAMPLES = 10000
    
    # ==================== Paths ====================
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    RESULTS_DIR = "./results"
    
    # ==================== Experiment Settings ====================
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("BEAM ALIGNMENT SYSTEM CONFIGURATION")
        print("=" * 60)
        print(f"\nAntenna Arrays:")
        print(f"  BS Transmit Antennas (NTX): {cls.NTX}")
        print(f"  UE Receive Antennas (NRX): {cls.NRX}")
        print(f"\nChannel (Sionna 3GPP TR 38.901 scenarios):")
        print(f"  Scenarios: {', '.join(cls.SCENARIOS)}")
        print(f"  O2I model (UMi/UMa): {cls.O2I_MODEL}")
        print(f"  Pathloss: {'on' if cls.ENABLE_PATHLOSS else 'off'}")
        print(f"  Shadow fading: {'on' if cls.ENABLE_SHADOW_FADING else 'off'}")
        print(f"  Distance: {cls.DISTANCE_RANGE_M[0]:.0f}-{cls.DISTANCE_RANGE_M[1]:.0f} m")
        print(f"  UE Speed: {cls.UE_SPEED_RANGE[0]:.0f}-{cls.UE_SPEED_RANGE[1]:.0f} m/s")
        print(f"  Time-varying channel: {'on' if getattr(cls, 'MOBILITY_ENABLE', False) else 'off'}")
        if getattr(cls, "MOBILITY_ENABLE", False):
            nts = getattr(cls, "MOBILITY_NUM_TIME_SAMPLES", None)
            nts_eff = (cls.T + 1) if nts is None else int(nts)
            print(f"  Channel time samples: {nts_eff}")
            fs = float(getattr(cls, "MOBILITY_SAMPLING_FREQUENCY_HZ", 1.0))
            print(f"  Channel sampling f_s: {fs/1e3:.1f} kHz")
        print(f"  Indoor probability: {cls.INDOOR_PROBABILITY:.2f}")
        print(f"  Carrier Frequency: {cls.CARRIER_FREQUENCY/1e9:.0f} GHz")
        print(f"  Wavelength: {cls.WAVELENGTH*1000:.2f} mm")
        print(f"  Channel gen device: {getattr(cls, 'CHANNEL_GENERATION_DEVICE', 'auto')}")
        print(
            f"  Train channels outside graph: {'on' if getattr(cls, 'TRAIN_CHANNELS_OUTSIDE_GRAPH', False) else 'off'}"
        )
        print(f"\nBeam Alignment:")
        print(f"  Sensing Steps (T): {cls.T}")
        print(f"  BS Codebook Size (NCB): {cls.NCB}")
        print(f"  Random sweep start: {cls.RANDOM_START}")
        if not cls.RANDOM_START:
            print(f"  Fixed sweep start index: {cls.START_BEAM_INDEX}")
        print(f"\nTraining:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        if cls.SNR_TRAIN_RANDOMIZE:
            print(f"  Training SNR: {cls.SNR_TRAIN_RANGE[0]:.1f}-{cls.SNR_TRAIN_RANGE[1]:.1f} dB (randomized)")
        else:
            print(f"  Training SNR: {cls.SNR_TRAIN} dB (fixed)")
        print(f"\nUE Controller (RNN):")
        print(f"  Type: {cls.RNN_TYPE}")
        print(f"  Layers: {getattr(cls, 'RNN_NUM_LAYERS', 2)}")
        print(f"  Hidden Size: {cls.RNN_HIDDEN_SIZE}")
        print(f"  Feedback Size (NFB): {cls.NUM_FEEDBACK}")
        print(f"  Beam index encoding: {getattr(cls, 'UE_BEAM_INDEX_ENCODING', 'scalar')}")
        print(f"  Include time feature: {getattr(cls, 'UE_INCLUDE_TIME_FEATURE', False)}")
        print(f"  Input layer norm: {getattr(cls, 'UE_INPUT_LAYER_NORM', False)}")
        print(f"  Output layer norm: {getattr(cls, 'UE_OUTPUT_LAYER_NORM', False)}")
        print(f"  UE dropout: {getattr(cls, 'UE_DROPOUT_RATE', 0.0)}")
        print(f"\nBS Feedback Network (N2):")
        print(f"  Hidden sizes: {getattr(cls, 'BS_FNN_HIDDEN_SIZES', (256, 256))}")
        print(f"  Activation: {getattr(cls, 'BS_FNN_ACTIVATION', 'gelu')}")
        print(f"  Layer norm: {getattr(cls, 'BS_FNN_LAYER_NORM', True)}")
        print(f"\nDataset:")
        print(f"  Training Samples: {cls.NUM_TRAIN_SAMPLES:,}")
        print(f"  Validation Samples: {cls.NUM_VAL_SAMPLES:,}")
        print(f"  Test Samples: {cls.NUM_TEST_SAMPLES:,}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()
