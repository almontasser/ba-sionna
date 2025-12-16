"""
Channel generation worker for multiprocessing.

Each worker process initializes its own TensorFlow and Sionna instance
to avoid race conditions and enable true parallel channel generation.
"""

import os
import numpy as np


def init_worker(config_dict):
    """Initialize TensorFlow and channel model in worker process."""
    global _worker_model, _worker_config
    
    # Suppress TF warnings in workers
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    
    # Limit GPU memory in workers to avoid OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Import after TF setup
    from channels.tr38901_scenario import SionnaScenarioChannelModel
    
    # Create channel model
    _worker_config = config_dict
    _worker_model = SionnaScenarioChannelModel(
        num_tx_antennas=config_dict['NTX'],
        num_rx_antennas=config_dict['NRX'],
        carrier_frequency=config_dict.get('CARRIER_FREQUENCY', 28e9),
        scenarios=config_dict.get('SCENARIOS', ['UMi', 'UMa', 'RMa']),
        distance_range_m=config_dict.get('DISTANCE_RANGE_M', (10.0, 200.0)),
        ue_speed_range=config_dict.get('UE_SPEED_RANGE', (0.0, 30.0)),
        indoor_probability=config_dict.get('INDOOR_PROBABILITY', 0.0),
        fft_size=config_dict.get('RESOURCE_GRID_FFT_SIZE', 64),
        subcarrier_spacing=config_dict.get('RESOURCE_GRID_SUBCARRIER_SPACING', 120e3),
        narrowband_method=config_dict.get('NARROWBAND_METHOD', 'center'),
        generation_device=config_dict.get('CHANNEL_GENERATION_DEVICE', 'cpu'),
    )


def generate_channel_batch(batch_size):
    """Generate a single batch of channels in worker process."""
    global _worker_model
    
    channels, _, _ = _worker_model.generate_channel(batch_size)
    # Convert to numpy for pickling back to main process
    return channels.numpy()


def generate_channels_multiprocess(config, num_batches, batch_size, num_workers=4):
    """
    Generate channel batches using multiple processes.
    
    Args:
        config: Config object with channel parameters
        num_batches: Number of channel batches to generate
        batch_size: Size of each batch
        num_workers: Number of worker processes
        
    Returns:
        List of channel tensors
    """
    import multiprocessing as mp
    from functools import partial
    
    # Extract config as dict for pickling
    config_dict = {
        'NTX': config.NTX,
        'NRX': config.NRX,
        'CARRIER_FREQUENCY': getattr(config, 'CARRIER_FREQUENCY', 28e9),
        'SCENARIOS': getattr(config, 'SCENARIOS', ['UMi', 'UMa', 'RMa']),
        'DISTANCE_RANGE_M': getattr(config, 'DISTANCE_RANGE_M', (10.0, 200.0)),
        'UE_SPEED_RANGE': getattr(config, 'UE_SPEED_RANGE', (0.0, 30.0)),
        'INDOOR_PROBABILITY': getattr(config, 'INDOOR_PROBABILITY', 0.0),
        'RESOURCE_GRID_FFT_SIZE': getattr(config, 'RESOURCE_GRID_FFT_SIZE', 64),
        'RESOURCE_GRID_SUBCARRIER_SPACING': getattr(config, 'RESOURCE_GRID_SUBCARRIER_SPACING', 120e3),
        'NARROWBAND_METHOD': getattr(config, 'NARROWBAND_METHOD', 'center'),
        'CHANNEL_GENERATION_DEVICE': getattr(config, 'CHANNEL_GENERATION_DEVICE', 'cpu'),
    }
    
    # Use 'spawn' to avoid fork issues with TensorFlow
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config_dict,)
    ) as pool:
        # Generate all batches in parallel
        results = pool.map(generate_channel_batch, [batch_size] * num_batches)
    
    # Convert numpy arrays back to TensorFlow tensors
    import tensorflow as tf
    return [tf.constant(r, dtype=tf.complex64) for r in results]
