"""
Training Script for mmWave Beam Alignment Model

This script trains an end-to-end beam alignment model for mmWave communication systems
using adaptive sensing and feedback mechanisms. The model learns to select optimal
beam pairs between a Base Station (BS) and User Equipment (UE) through sequential
sensing steps guided by a recurrent neural network.

This implementation trains the full end-to-end beam alignment pipeline
(paper "C3" variant): UE RNN + BS FNN + learnable BS codebook.

Key paper-style parameters (Section IV): NTX=32, NRX=16, NCB=8,
batch_size=256, training SNR=10 dB, 2-layer GRU.

Note: model size is configurable; with the current default config the C3 model
is ~0.7M parameters (and can be reduced/increased via Config).

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
import os
from datetime import datetime
from tqdm import tqdm
import io
import contextlib
import threading
import queue

from device_setup import setup_device, print_device_info
from config import Config
from models.beam_alignment import BeamAlignmentModel
from checkpoint_utils import check_checkpoint_compatibility
from training.lr_schedule import WarmupThenDecay
from training.steps import sample_snr, train_step, validate


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
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(run_dir)

    # Log the full config once so runs are self-describing in TensorBoard.
    cfg_buf = io.StringIO()
    with contextlib.redirect_stdout(cfg_buf):
        config.print_config()
    with summary_writer.as_default():
        tf.summary.text("run/config", tf.constant(cfg_buf.getvalue()), step=0)

    print(f"\nTensorBoard run: {run_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    with tf.device(device_string):
        # Create model
        print(f"\nCreating model on {device_name}...")
        model = create_model(config)

        # Build the model by running a dummy forward pass before restoring checkpoint.
        print("Building model...")
        _ = model(batch_size=config.BATCH_SIZE, snr_db=config.SNR_TRAIN, training=False)

        # Setup checkpoint manager (model weights only).
        #
        # This is intentionally model-only: optimizer state is fragile across
        # architecture changes and can cause hard failures when restoring.
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5
        )

        # Restore from checkpoint if available and compatible.
        start_epoch = 0
        if checkpoint_manager.latest_checkpoint:
            ckpt_path = checkpoint_manager.latest_checkpoint
            compat = check_checkpoint_compatibility(
                model,
                ckpt_path,
                require_all_trainable=True,
                require_ue_rnn_kernel=True,
            )
            if compat.ok:
                try:
                    checkpoint.restore(ckpt_path).expect_partial()
                    print(f"✓ Restored model weights from {ckpt_path}")
                except (ValueError, tf.errors.InvalidArgumentError) as e:
                    print(
                        "WARNING: Found a checkpoint but restore failed due to incompatibility.\n"
                        "Starting training from scratch.\n"
                        f"Restore error: {e}"
                    )
                    # Avoid overwriting incompatible checkpoints by switching
                    # to a fresh directory.
                    fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                    print(f"Using fresh checkpoint dir: {fresh_dir}")
                    os.makedirs(fresh_dir, exist_ok=True)
                    checkpoint_dir = fresh_dir
                    checkpoint_manager = tf.train.CheckpointManager(
                        checkpoint, checkpoint_dir, max_to_keep=5
                    )
            else:
                print(
                    "WARNING: Found a checkpoint but it does not match the current model.\n"
                    "Starting training from scratch.\n"
                    "Details:\n"
                    f"{compat.summary()}\n"
                    "Tip: use a fresh --checkpoint_dir (or delete old checkpoints) after changing architecture."
                )
                fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                print(f"Using fresh checkpoint dir: {fresh_dir}")
                os.makedirs(fresh_dir, exist_ok=True)
                checkpoint_dir = fresh_dir
                checkpoint_manager = tf.train.CheckpointManager(
                    checkpoint, checkpoint_dir, max_to_keep=5
                )
        else:
            print("Starting training from scratch")

        # Create optimizer with learning rate schedule
        steps_per_epoch = max(1, config.NUM_TRAIN_SAMPLES // config.BATCH_SIZE)
        # Warm-up + decay schedule
        base_lr = config.LEARNING_RATE
        warmup_epochs = getattr(config, "LR_WARMUP_EPOCHS", 0) or 0
        decay_steps = max(1, config.LEARNING_RATE_DECAY_STEPS * steps_per_epoch)

        warmup_steps = warmup_epochs * steps_per_epoch
        lr_schedule = WarmupThenDecay(
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_rate=config.LEARNING_RATE_DECAY,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
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
            
            # Use channel caching for faster training
            if getattr(config, "TRAIN_CHANNELS_OUTSIDE_GRAPH", False):
                cache_size = getattr(config, "CHANNEL_CACHE_SIZE", 0)
                
                if cache_size > 0:
                    # Pre-generate channel cache at epoch start
                    if epoch == start_epoch or not hasattr(train, '_channel_cache'):
                        print(f"Pre-generating {cache_size} channel batches for cache...")
                        train._channel_cache = []
                        for i in tqdm(range(cache_size), desc="Caching channels"):
                            channels, _, _ = model.generate_channels(config.BATCH_SIZE)
                            train._channel_cache.append(channels)
                        print(f"✓ Channel cache ready ({cache_size} batches)")
                    
                    # Training using cached channels (random sampling)
                    pbar = tqdm(range(steps_per_epoch), desc="Training")
                    for step in pbar:
                        # Random sample from cache
                        cache_idx = tf.random.uniform([], 0, cache_size, dtype=tf.int32).numpy()
                        channels = train._channel_cache[cache_idx]
                        snr_db = sample_snr(config)
                        
                        loss, bf_gain_db, grad_norm = train_step(
                            model,
                            optimizer,
                            config.BATCH_SIZE,
                            snr_db,
                            channels=channels,
                        )
                        
                        epoch_loss += loss.numpy()
                        epoch_bf_gain += bf_gain_db.numpy()
                        global_step += 1
                        
                        # Update progress bar
                        gain_norm = -loss
                        pbar_dict = {
                            'loss': f'{loss.numpy():.4f}',
                            'gain_norm': f'{gain_norm.numpy():.4f}',
                            'BF_gain': f'{bf_gain_db.numpy():.2f} dB',
                            'grad_norm': f'{grad_norm.numpy():.3f}'
                        }
                        pbar.set_postfix(pbar_dict)
                        
                        # Log to TensorBoard
                        if step % 10 == 0:
                            with summary_writer.as_default():
                                tf.summary.scalar("train/loss", loss, step=global_step)
                                tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                                tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                                tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                                tf.summary.scalar(
                                    "train/learning_rate", optimizer.learning_rate, step=global_step
                                )
                else:
                    # No caching - generate fresh channels each iteration (slow but max diversity)
                    pbar = tqdm(range(steps_per_epoch), desc="Training")
                    for step in pbar:
                        channels, _, _ = model.generate_channels(config.BATCH_SIZE)
                        snr_db = sample_snr(config)
                        
                        loss, bf_gain_db, grad_norm = train_step(
                            model,
                            optimizer,
                            config.BATCH_SIZE,
                            snr_db,
                            channels=channels,
                        )
                        
                        epoch_loss += loss.numpy()
                        epoch_bf_gain += bf_gain_db.numpy()
                        global_step += 1
                        
                        gain_norm = -loss
                        pbar_dict = {
                            'loss': f'{loss.numpy():.4f}',
                            'gain_norm': f'{gain_norm.numpy():.4f}',
                            'BF_gain': f'{bf_gain_db.numpy():.2f} dB',
                            'grad_norm': f'{grad_norm.numpy():.3f}'
                        }
                        pbar.set_postfix(pbar_dict)
                        
                        if step % 10 == 0:
                            with summary_writer.as_default():
                                tf.summary.scalar("train/loss", loss, step=global_step)
                                tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                                tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                                tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                                tf.summary.scalar(
                                    "train/learning_rate", optimizer.learning_rate, step=global_step
                                )
            else:
                # Fallback: generate channels inside the training step (legacy mode)
                pbar = tqdm(range(steps_per_epoch), desc="Training")
                for step in pbar:
                    snr_db = sample_snr(config)
                    loss, bf_gain_db, grad_norm = train_step(
                        model,
                        optimizer,
                        config.BATCH_SIZE,
                        snr_db,
                        channels=None,
                    )
                    
                    epoch_loss += loss.numpy()
                    epoch_bf_gain += bf_gain_db.numpy()
                    global_step += 1
                    
                    gain_norm = -loss
                    pbar_dict = {
                        'loss': f'{loss.numpy():.4f}',
                        'gain_norm': f'{gain_norm.numpy():.4f}',
                        'BF_gain': f'{bf_gain_db.numpy():.2f} dB',
                        'grad_norm': f'{grad_norm.numpy():.3f}'
                    }
                    pbar.set_postfix(pbar_dict)
                    
                    if step % 10 == 0:
                        with summary_writer.as_default():
                            tf.summary.scalar("train/loss", loss, step=global_step)
                            tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                            tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                            tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                            tf.summary.scalar(
                                "train/learning_rate", optimizer.learning_rate, step=global_step
                            )
            
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
            with summary_writer.as_default():
                tf.summary.scalar("val/loss", val_metrics["val_loss"], step=global_step)
                tf.summary.scalar("val/bf_gain_db", val_metrics["mean_bf_gain_db"], step=global_step)
                tf.summary.scalar(
                    "val/satisfaction_prob", val_metrics["satisfaction_prob"], step=global_step
                )
            
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
    print(f"  (this run: {run_dir})")


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
        '--require_gpu',
        action='store_true',
        help='Fail fast if no GPU is visible to TensorFlow',
    )
    parser.add_argument(
        '--channel_gen_device',
        type=str,
        default=None,
        choices=['auto', 'cpu', 'gpu'],
        help='Channel generation device placement (overrides Config.CHANNEL_GENERATION_DEVICE)',
    )
    parser.add_argument(
        '--train_channels_outside_graph',
        type=int,
        default=None,
        choices=[0, 1],
        help='If 1, generate channels outside @tf.function (better GPU placement); overrides Config.TRAIN_CHANNELS_OUTSIDE_GRAPH',
    )
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
    parser.add_argument(
        '--xla_jit',
        type=int,
        default=None,
        choices=[0, 1],
        help='Enable XLA JIT compilation (1=on, 0=off). Default: on for GPU. Disable if XLA causes errors.',
    )
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
    if args.channel_gen_device is not None:
        Config.CHANNEL_GENERATION_DEVICE = args.channel_gen_device
    if args.train_channels_outside_graph is not None:
        Config.TRAIN_CHANNELS_OUTSIDE_GRAPH = bool(int(args.train_channels_outside_graph))
    if args.snr_train_range is not None:
        try:
            low, high = args.snr_train_range.split(',')
            Config.SNR_TRAIN_RANGE = (float(low), float(high))
        except Exception as e:
            raise ValueError(f"Invalid --snr_train_range '{args.snr_train_range}'. Expected format: low,high") from e
    if args.xla_jit is not None:
        Config.XLA_JIT_COMPILE = bool(args.xla_jit)
    
    if args.test_mode:
        Config.EPOCHS = 1
        Config.NUM_TRAIN_SAMPLES = 1000
        Config.NUM_VAL_SAMPLES = 200
        print("Running in TEST MODE (reduced dataset)")
    
    # Set checkpoint directory (C3-only) if not explicitly provided
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = f"./checkpoints_C3_T{Config.T}"
    
    if args.require_gpu:
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError(
                "No GPU visible to TensorFlow, but --require_gpu was set. "
                "Check your CUDA/TensorFlow install and CUDA_VISIBLE_DEVICES."
            )

    # Run training
    train(Config, checkpoint_dir=checkpoint_dir, log_dir=args.log_dir)
