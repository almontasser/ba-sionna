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

2. Channel Model Parameters (Sionna CDL):
   - CARRIER_FREQUENCY: 28 GHz (mmWave band)
   - 3GPP TR 38.901 CDL profiles (A–E)

3. Beam Alignment Parameters:
   - T: Number of sensing steps (8)
   - NCB: BS codebook size (8 beams)
   - Start beam selection: fixed or random

4. Training Parameters:
   - BATCH_SIZE: 256 (set for 15 GB VRAM)
   - EPOCHS: 100
   - LEARNING_RATE: 0.001 with exponential decay
   - SNR_TRAIN: 10 dB training SNR
   - SNR_TEST_RANGE: -10 to 20 dB for evaluation

5. Neural Network Architecture:
   - RNN_TYPE: "GRU" (2-layer as per paper)
   - RNN_HIDDEN_SIZE: 384 (for ~500K parameters)
   - NUM_FEEDBACK: 16 feedback values

6. Experiment Variants:
   - C2: Fixed starting beam from codebook
   - C3: Random starting beam (used in paper)

Usage:
    As a class with static attributes:
        >>> from config import Config
        >>> print(Config.NTX, Config.NRX)
        32 16
        >>> Config.print_config()  # Print all settings
    
    For custom experiments:
        >>> custom_config = Config.get_config_for_experiment(
        ...     T=16, NCB=16, SNR=15.0
        ... )
    
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
    
    # ==================== Channel Model Parameters (Sionna CDL only) ====================
    CARRIER_FREQUENCY = 28e9  # 28 GHz (mmWave)
    WAVELENGTH = 3e8 / CARRIER_FREQUENCY  # Speed of light / frequency
    ANTENNA_SPACING = WAVELENGTH / 2  # Half-wavelength spacing for ULA
    
    # Domain Randomization Parameters (for robust training)
    CDL_MODELS = ["A", "B", "C", "D", "E"]  # All 3GPP CDL profiles
    DELAY_SPREAD_RANGE = (10e-9, 300e-9)  # 10ns to 300ns (typical mmWave range)
    UE_SPEED_RANGE = (0.0, 30.0)  # 0 to 30 m/s (0 to 108 km/h)
    # OFDM resource grid for Sionna channel generation
    RESOURCE_GRID_FFT_SIZE = 64
    RESOURCE_GRID_NUM_OFDM_SYMBOLS = 1
    RESOURCE_GRID_SUBCARRIER_SPACING = 120e3  # Hz
    
    # ==================== Beam Alignment Parameters ====================
    T = 16  # Number of sensing steps (Paper uses T=16 for Figure 4)
    NCB = 8  # Codebook size at BS (number of beams in learned codebook)
    
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
    # How to reduce the CDL CIR/frequency response to a narrowband H used in y_t.
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
    RNN_HIDDEN_SIZE = 384  # Matches paper's ~500K total parameters (was 256)
    NUM_FEEDBACK = 16  # Number of feedback values (NFB) (per arXiv paper)
    
    # C1 Scheme Parameters
    C1_CE_LOSS_WEIGHT = 0.1  # Weight for cross-entropy auxiliary loss (paper doesn't specify, using 0.1)
    
    # ==================== Model Architecture ====================
    USE_BATCH_NORM = False
    DROPOUT_RATE = 0.0  # Set to 0.0 to disable dropout
    
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
    
    # Experiment variants
    VARIANTS = {
        "C2": {  # Fixed starting beam
            "description": "Fixed starting beam from codebook",
            "random_start": False,
            "start_index": 0
        },
        "C3": {  # Random starting beam
            "description": "Random starting beam from codebook",
            "random_start": True,
            "start_index": None
        }
    }
    
    # ==================== Baseline Configurations ====================
    EXHAUSTIVE_SEARCH_GRID_SIZE = 32  # Grid size for exhaustive search
    
    @classmethod
    def get_noise_power(cls, snr_db):
        """
        Calculate noise power from SNR in dB.
        
        Args:
            snr_db: SNR per antenna in decibels
            
        Returns:
            noise_power: Noise power (linear scale)
        """
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1.0 / (cls.NRX * snr_linear)
        return noise_power
    
    @classmethod
    def get_config_for_experiment(cls, T=None, NCB=None, SNR=None):
        """
        Get a modified config for specific experiments.
        
        Args:
            T: Number of sensing steps (overrides default)
            NCB: Codebook size (overrides default)
            SNR: SNR in dB (overrides default)
            
        Returns:
            Modified config dictionary
        """
        config_dict = {
            'NTX': cls.NTX,
            'NRX': cls.NRX,
            'T': T if T is not None else cls.T,
            'NCB': NCB if NCB is not None else cls.NCB,
            'SNR': SNR if SNR is not None else cls.SNR_TRAIN,
            'RNN_HIDDEN_SIZE': cls.RNN_HIDDEN_SIZE,
            'NUM_FEEDBACK': cls.NUM_FEEDBACK,
        }
        return config_dict
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("BEAM ALIGNMENT SYSTEM CONFIGURATION")
        print("=" * 60)
        print(f"\nAntenna Arrays:")
        print(f"  BS Transmit Antennas (NTX): {cls.NTX}")
        print(f"  UE Receive Antennas (NRX): {cls.NRX}")
        print(f"\nChannel (Sionna 3GPP TR 38.901 CDL):")
        print(f"  CDL Profiles: {', '.join(['CDL-' + m for m in cls.CDL_MODELS])}")
        print(f"  Delay Spread: {cls.DELAY_SPREAD_RANGE[0]*1e9:.0f}-{cls.DELAY_SPREAD_RANGE[1]*1e9:.0f} ns")
        print(f"  UE Speed: {cls.UE_SPEED_RANGE[0]:.0f}-{cls.UE_SPEED_RANGE[1]:.0f} m/s")
        print(f"  Carrier Frequency: {cls.CARRIER_FREQUENCY/1e9:.0f} GHz")
        print(f"  Wavelength: {cls.WAVELENGTH*1000:.2f} mm")
        print(f"\nBeam Alignment:")
        print(f"  Sensing Steps (T): {cls.T}")
        print(f"  BS Codebook Size (NCB): {cls.NCB}")
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
        print(f"  Hidden Size: {cls.RNN_HIDDEN_SIZE}")
        print(f"  Feedback Size (NFB): {cls.NUM_FEEDBACK}")
        print(f"\nDataset:")
        print(f"  Training Samples: {cls.NUM_TRAIN_SAMPLES:,}")
        print(f"  Validation Samples: {cls.NUM_VAL_SAMPLES:,}")
        print(f"  Test Samples: {cls.NUM_TEST_SAMPLES:,}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()
