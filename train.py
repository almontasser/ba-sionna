"""
Training Script for mmWave Beam Alignment Model

This script trains an end-to-end beam alignment model for mmWave communication systems
using adaptive sensing and feedback mechanisms. The model learns to select optimal
beam pairs between a Base Station (BS) and User Equipment (UE) through sequential
sensing steps guided by a recurrent neural network.

This implementation trains the full end-to-end beam alignment pipeline
(paper "C3" variant): UE RNN + BS FNN + learnable BS codebook.

Key paper parameters (Section IV): NTX=32, NRX=16, L=3 paths, NCB=8, 
batch_size=256 (tuned for 15 GB VRAM), training SNR=10dB, ~500K parameters, 2-layer GRU

Usage:
    Basic training:
        $ python train.py
    
    Custom configuration:
        $ python train.py --epochs 200 --batch_size 512 --lr 0.0005
    
    Resume from checkpoint:
        $ python train.py --checkpoint_dir ./checkpoints/experiment_1
    
    Test mode (reduced dataset):
        $ python train.py --test_mode

Command-line Arguments:
    --epochs: Number of training epochs (default: Config.EPOCHS)
    --batch_size: Batch size for training (default: Config.BATCH_SIZE)
    --lr: Learning rate (default: Config.LEARNING_RATE)
    --checkpoint_dir: Directory to save checkpoints (default: Config.CHECKPOINT_DIR)
    --log_dir: Directory for TensorBoard logs (default: Config.LOG_DIR)
    --test_mode: Run in test mode with reduced dataset (1 epoch, 1000 samples)

Outputs:
    - Checkpoints saved to checkpoint_dir every 10 epochs and when validation improves
    - TensorBoard logs saved to log_dir for training visualization
    - Best model selected based on validation beamforming gain

Monitoring:
    View training progress with TensorBoard:
        $ tensorboard --logdir ./logs
"""

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from device_setup import setup_device, print_device_info
from config import Config
from models.beam_alignment import BeamAlignmentModel
from metrics import BeamAlignmentMetrics, compute_loss
from utils import compute_beamforming_gain_db


def sample_snr(config):
    """
    Sample a random SNR value for domain randomization.
    
    Args:
        config: Configuration object with SNR_TRAIN_RANDOMIZE and SNR_TRAIN_RANGE
        
    Returns:
        SNR value in dB (float32 scalar tensor)
    """
    if config.SNR_TRAIN_RANDOMIZE:
        # Sample from uniform distribution over training range
        snr_db = tf.random.uniform(
            [], 
            minval=config.SNR_TRAIN_RANGE[0], 
            maxval=config.SNR_TRAIN_RANGE[1],
            dtype=tf.float32
        )
    else:
        # Use fixed training SNR
        snr_db = tf.constant(config.SNR_TRAIN, dtype=tf.float32)
    
    return snr_db


def create_model(config):
    """
    Create and initialize a BeamAlignmentModel from configuration.
    
    This function instantiates the complete end-to-end beam alignment system,
    including the BS controller, UE controller, and channel model components.
    
    Args:
        config: Configuration object (Config class) containing:
            - NTX: Number of BS transmit antennas
            - NRX: Number of UE receive antennas
            - NCB: BS codebook size
            - T: Number of sensing steps
            - RNN_HIDDEN_SIZE: UE RNN hidden state size
            - RNN_TYPE: UE RNN type ("GRU" or "LSTM")
            - NUM_FEEDBACK: Number of feedback values from UE to BS
    Returns:
        BeamAlignmentModel: Initialized beam alignment model ready for training
    
    Example:
        >>> from config import Config
        >>> model = create_model(Config)
        >>> # Model is now ready for training
        >>> results = model(batch_size=32, snr_db=10.0, training=True)
    """
    model = BeamAlignmentModel(
        num_tx_antennas=config.NTX,
        num_rx_antennas=config.NRX,
        codebook_size=config.NCB,
        num_sensing_steps=config.T,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_type=config.RNN_TYPE,
        num_feedback=config.NUM_FEEDBACK,
        start_beam_index=getattr(config, "START_BEAM_INDEX", 0),
        random_start=getattr(config, "RANDOM_START", True),
        carrier_frequency=config.CARRIER_FREQUENCY,
        scenarios=config.SCENARIOS,
        o2i_model=getattr(config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(config, "BS_HEIGHT_RMA_M", 35.0),
    )

    return model



@tf.function(reduce_retracing=True)
def train_step(model, optimizer, batch_size, snr_db):
    """
    Execute one training step with domain randomization.
    
    Args:
        model: BeamAlignmentModel
        optimizer: TensorFlow optimizer
        batch_size: Batch size
        snr_db: SNR in dB (can be scalar or tensor for randomization)
    Returns:
        loss: Total loss
        beamforming_gain_db: Mean BF gain in dB
        gradient_norm: Gradient norm
        
    Note:
        Domain randomization is applied via:
        1. Random scenario selection (UMi/UMa/RMa) and topology sampling
           (in channel_model.generate_channel)
        2. Random SNR per batch (if enabled via Config.SNR_TRAIN_RANGE)
    """
    with tf.GradientTape() as tape:
        # Forward pass
        results = model(batch_size=batch_size, snr_db=snr_db, training=True)

        loss = compute_loss(
            results['beamforming_gain'],
            results['channels'],
            loss_type=getattr(Config, "LOSS_TYPE", "paper"),
        )
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Compute gradient norm BEFORE clipping (important diagnostic)
    gradient_norm = tf.linalg.global_norm(gradients)
    
    # Clip gradients to prevent explosion
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Compute mean BF gain in dB for logging
    bf_gain_db = tf.reduce_mean(
        compute_beamforming_gain_db(
            results['channels'],
            results['final_tx_beams'],
            results['final_rx_beams']
        )
    )

    return loss, bf_gain_db, gradient_norm


@tf.function(reduce_retracing=True)
def _validate_step(model, batch_size, snr_db):
    """Single validation step (graph-compiled for speed)."""
    results = model(batch_size=batch_size, snr_db=snr_db, training=False)
    loss = compute_loss(
        results['beamforming_gain'],
        results['channels'],
        loss_type=getattr(Config, "LOSS_TYPE", "paper"),
    )
    
    # Compute beamforming gain in dB for metrics
    bf_gain_db = compute_beamforming_gain_db(
        results['channels'],
        results['final_tx_beams'],
        results['final_rx_beams']
    )
    
    return loss, bf_gain_db


def validate(model, num_val_batches, batch_size, snr_db, target_snr_db):
    """
    Validate the model.
    
    Args:
        model: BeamAlignmentModel
        num_val_batches: Number of validation batches
        batch_size: Batch size
        snr_db: SNR in dB
        target_snr_db: Target SNR for satisfaction probability
        
    Returns:
        Dictionary with validation metrics
    """
    # OPTIMIZATION: Use fewer, larger batches for faster validation
    # Instead of many small batches, use 2-3 large batches
    num_val_batches = max(2, min(num_val_batches, 3))
    val_batch_size = batch_size * 2  # Use larger batches for validation
    
    total_loss = 0.0
    all_bf_gains_db = []
    
    for _ in range(num_val_batches):
        loss, bf_gain_db = _validate_step(model, val_batch_size, snr_db)
        total_loss += loss.numpy()
        all_bf_gains_db.append(bf_gain_db)
    
    # Combine all beamforming gains
    all_bf_gains_db = tf.concat(all_bf_gains_db, axis=0)
    
    # Compute metrics
    mean_bf_gain = tf.reduce_mean(all_bf_gains_db)
    std_bf_gain = tf.math.reduce_std(all_bf_gains_db)
    
    # Satisfaction probability (paper Eq. (4)–(6)) uses post-combining receive SNR:
    #   SNR_RX(dB) = 10log10(|w^H H f|^2 / sigma_n^2)
    snr_linear = 10.0 ** (tf.cast(snr_db, tf.float32) / 10.0)
    # Paper Eq. (4): per-antenna SNR_ANT = 1/sigma_n^2 => sigma_n^2 = 1/SNR_ANT
    noise_power = 1.0 / snr_linear  # sigma_n^2 in y_t
    noise_power_db = 10.0 * tf.math.log(noise_power + 1e-20) / tf.math.log(10.0)
    snr_rx_db = all_bf_gains_db - tf.cast(noise_power_db, all_bf_gains_db.dtype)
    satisfaction_prob = tf.reduce_mean(tf.cast(snr_rx_db >= target_snr_db, tf.float32))
    
    metric_results = {
        'val_loss': total_loss / num_val_batches,
        'mean_bf_gain_db': mean_bf_gain.numpy(),
        'std_bf_gain_db': std_bf_gain.numpy(),
        'satisfaction_prob': satisfaction_prob.numpy()
    }
    
    return metric_results


def train(config, checkpoint_dir=None, log_dir=None):
    """
    Main training loop.
    
    Args:
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    print("=" * 80)
    print("BEAM ALIGNMENT TRAINING")
    print("=" * 80)
    
    # Setup device
    print("\nDevice Setup:")
    print_device_info()
    device_string, device_name = setup_device(verbose=False)
    
    # Enable mixed precision training for better GPU performance (~2x speedup on modern GPUs)
    if 'GPU' in device_string:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision training enabled (float16) for faster GPU training\n")
    
    # Print configuration
    print("\n")
    config.print_config()
    
    # Create directories
    if checkpoint_dir is None:
        checkpoint_dir = f"{config.CHECKPOINT_DIR}_C3_T{config.T}"
    if log_dir is None:
        log_dir = config.LOG_DIR
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, f'train_{timestamp}')
    val_log_dir = os.path.join(log_dir, f'val_{timestamp}')
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    print(f"\nTensorBoard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    with tf.device(device_string):
        # Create model
        print(f"\nCreating model on {device_name}...")
        model = create_model(config)
        
        # Create optimizer with learning rate schedule
        steps_per_epoch = max(1, config.NUM_TRAIN_SAMPLES // config.BATCH_SIZE)
        # Warm-up + decay schedule
        base_lr = config.LEARNING_RATE
        warmup_epochs = getattr(config, "LR_WARMUP_EPOCHS", 0) or 0
        decay_steps = max(1, config.LEARNING_RATE_DECAY_STEPS * steps_per_epoch)

        class WarmupThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, base_lr, warmup_steps, decay_steps, decay_rate):
                self.base_lr = base_lr
                self.warmup_steps = warmup_steps
                self.decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=base_lr,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True,
                )

            def __call__(self, step):
                if self.warmup_steps > 0:
                    warm_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
                    lr = tf.where(step < self.warmup_steps, warm_lr, self.decay(step - self.warmup_steps))
                else:
                    lr = self.decay(step)
                return lr

        warmup_steps = warmup_epochs * steps_per_epoch
        lr_schedule = WarmupThenDecay(
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_rate=config.LEARNING_RATE_DECAY,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Setup checkpoint manager
        # NOTE: We only save model and optimizer state, not the epoch number
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5
        )
        
        # Build the model by running a dummy forward pass before restoring checkpoint
        print("Building model...")
        _ = model(batch_size=config.BATCH_SIZE, snr_db=config.SNR_TRAIN, training=False)
        
        # CRITICAL FIX: Run a dummy training step to initialize optimizer variables!
        # The checkpoint contains optimizer variables (momentum, etc.).
        # If we don't use the optimizer once, these variables don't exist in the current object,
        # so restore() fails to load them (and thus fails to load the model weights linked to them).
        print("Initializing optimizer variables (dummy step)...")
        # Use a small batch for speed
        train_step(model, optimizer, batch_size=16, snr_db=config.SNR_TRAIN)
        
        # DEBUG: Print first variable value before restore
        if len(model.trainable_variables) > 0:
            print(f"DEBUG: First variable before restore: {model.trainable_variables[0].name}")
            print(f"DEBUG: Value sample: {model.trainable_variables[0].numpy().flatten()[:5]}")
        
        # Restore from checkpoint if available
        # NOTE: Checkpoint number (e.g., ckpt-30) is NOT the epoch number!
        # The checkpoint number is just an internal counter managed by CheckpointManager.
        # Training always resumes from epoch 0 with restored weights.
        start_epoch = 0
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            
            # STRICT CHECK: Ensure model variables are loaded
            # assert_existing_objects_matched() ensures that for every Python object 
            # in the checkpoint (model, optimizer), values are found in the checkpoint file.
            try:
                status.assert_existing_objects_matched()
                print("✓ Checkpoint objects matched successfully")
            except AssertionError as e:
                print(f"WARNING: Checkpoint match error: {e}")
                print("Continuing but be warned: some variables might not have restored.")
            
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
            
            # DEBUG: Print first variable value after restore
            if len(model.trainable_variables) > 0:
                print(f"DEBUG: First variable after restore: {model.trainable_variables[0].name}")
                print(f"DEBUG: Value sample: {model.trainable_variables[0].numpy().flatten()[:5]}")
            
            print(f"Resuming training from epoch 1 (with restored weights)")
        else:
            print("Starting training from scratch")
        
        # Training loop
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        steps_per_epoch = max(1, config.NUM_TRAIN_SAMPLES // config.BATCH_SIZE)
        val_batches = max(1, config.NUM_VAL_SAMPLES // config.BATCH_SIZE)
        
        global_step = start_epoch * steps_per_epoch
        best_val_bf_gain = -float('inf')
        
        for epoch in range(start_epoch, config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
            print("-" * 80)
            
            # Training
            epoch_loss = 0.0
            epoch_bf_gain = 0.0
            
            pbar = tqdm(range(steps_per_epoch), desc="Training")
            for step in pbar:
                # Sample SNR for this batch (domain randomization)
                snr_db = sample_snr(config)
                
                loss, bf_gain_db, grad_norm = train_step(model, optimizer, config.BATCH_SIZE, snr_db)
                
                epoch_loss += loss.numpy()
                epoch_bf_gain += bf_gain_db.numpy()
                global_step += 1
                
                # Update progress bar
                pbar_dict = {
                    'loss': f'{loss.numpy():.4f}',
                    'BF_gain': f'{bf_gain_db.numpy():.2f} dB',
                    'grad_norm': f'{grad_norm.numpy():.3f}'
                }
                pbar.set_postfix(pbar_dict)
                
                # Log to TensorBoard
                if step % 10 == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=global_step)
                        tf.summary.scalar('bf_gain_db', bf_gain_db, step=global_step)
                        tf.summary.scalar('gradient_norm', grad_norm, step=global_step)
                        tf.summary.scalar('learning_rate', optimizer.learning_rate, step=global_step)
            
            # Epoch statistics
            avg_loss = epoch_loss / steps_per_epoch
            avg_bf_gain = epoch_bf_gain / steps_per_epoch
            
            print(f"\nTraining - Loss: {avg_loss:.4f}, BF Gain: {avg_bf_gain:.2f} dB")
            
            # Validation
            print("Validating...")
            val_metrics = validate(
                model, val_batches, config.BATCH_SIZE, 
                config.SNR_TRAIN, config.SNR_TARGET
            )
            
            print(f"Validation - Loss: {val_metrics['val_loss']:.4f}")
            print(f"             BF Gain: {val_metrics['mean_bf_gain_db']:.2f} ± {val_metrics['std_bf_gain_db']:.2f} dB")
            print(f"             Satisfaction Prob: {val_metrics['satisfaction_prob']:.3f}")
            
            # Log to TensorBoard
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_metrics['val_loss'], step=epoch)
                tf.summary.scalar('bf_gain_db', val_metrics['mean_bf_gain_db'], step=epoch)
                tf.summary.scalar('satisfaction_prob', val_metrics['satisfaction_prob'], step=epoch)
            
            # Save checkpoint
            if val_metrics['mean_bf_gain_db'] > best_val_bf_gain:
                best_val_bf_gain = val_metrics['mean_bf_gain_db']
                save_path = checkpoint_manager.save()
                print(f"✓ New best model! Saved checkpoint: {save_path}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = checkpoint_manager.save()
                print(f"Saved periodic checkpoint: {save_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation BF gain: {best_val_bf_gain:.2f} dB")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train beam alignment model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_sensing_steps', '-T', type=int, default=None, 
                       help='Number of sensing steps (T). Default: Config.T (16)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument(
        '--scenarios',
        type=str,
        default=None,
        help='Comma-separated list of scenarios to use (e.g., "UMi,UMa,RMa")',
    )
    parser.add_argument('--target_snr', type=float, default=None,
                       help='Target SNR (dB) for satisfaction probability metrics')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                       help='Number of warm-up epochs with linear LR ramp (0 = no warm-up)')
    parser.add_argument('--snr_train_range', type=str, default=None,
                       help='Training SNR range "low,high" in dB (e.g., "0,10")')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (1 epoch)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.lr_warmup_epochs is not None:
        Config.LR_WARMUP_EPOCHS = args.lr_warmup_epochs
    if args.num_sensing_steps is not None:
        Config.T = args.num_sensing_steps
    if args.scenarios is not None:
        Config.SCENARIOS = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    if args.target_snr is not None:
        Config.SNR_TARGET = args.target_snr
    if args.snr_train_range is not None:
        try:
            low, high = args.snr_train_range.split(',')
            Config.SNR_TRAIN_RANGE = (float(low), float(high))
        except Exception as e:
            raise ValueError(f"Invalid --snr_train_range '{args.snr_train_range}'. Expected format: low,high") from e
    
    if args.test_mode:
        Config.EPOCHS = 1
        Config.NUM_TRAIN_SAMPLES = 1000
        Config.NUM_VAL_SAMPLES = 200
        print("Running in TEST MODE (reduced dataset)")
    
    # Set checkpoint directory (C3-only) if not explicitly provided
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = f"./checkpoints_C3_T{Config.T}"
    
    # Run training
    train(Config, checkpoint_dir=checkpoint_dir, log_dir=args.log_dir)
