"""
Paper Figure Reproduction Script
Generates correct implementations of figures 4 and 5 from the paper:
- Figure 4: C1, C2, C3 comparison vs SNR
- Figure 5: C2, C3 comparison vs T (sensing steps)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

from device_setup import setup_device
from config import Config
from models.beam_alignment import BeamAlignmentModel
from metrics import BeamAlignmentMetrics
from utils import compute_beamforming_gain_db


# ==================== Model Loading and Evaluation ====================

def evaluate_at_snr(model, snr_db, num_samples, batch_size, target_snr_db):
    """
    Evaluate a model at specific SNR.
    
    Args:
        model: BeamAlignmentModel instance
        snr_db: SNR per antenna in dB
        num_samples: Number of samples to evaluate
        batch_size: Batch size
        target_snr_db: Target SNR for satisfaction probability
        
    Returns:
        Dictionary with metrics
    """
    metrics = BeamAlignmentMetrics(target_snr_db=target_snr_db)
    num_batches = max(1, num_samples // batch_size)
    
    for _ in range(num_batches):
        results = model(batch_size=batch_size, snr_db=snr_db, training=False)
        metrics.update(
            results['channels'],
            results['final_tx_beams'],
            results['final_rx_beams']
        )
    
    return metrics.result()


# ==================== Figure Generation Functions ====================

def generate_figure_4_c1_c2_c3(config, output_dir='./results', num_samples=2000):
    """
    Figure 4: BF gain and Satisfaction probability vs SNR
    Compares: C1, C2, C3 (proper paper definitions)
    
    This function properly implements the three schemes from the paper:
    - C1: Fixed codebook + Fixed start
    - C2: Learnable codebook + Fixed start
    - C3: Learnable codebook + Random start
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 4: C1 vs C2 vs C3")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # SNR range
    snr_range = np.arange(-15, 26, 5)
    batch_size = 256
    target_snr_db = 20.0
    
    # Storage
    c1_results = {'bf_gain': [], 'sat_prob': []}
    c2_results = {'bf_gain': [], 'sat_prob': []}
    c3_results = {'bf_gain': [], 'sat_prob': []}
    
    # Helper function to load model with scheme
    def load_scheme_model(scheme, checkpoint_subdir):
        """Load a model with the correct scheme configuration."""

        print(f"  Creating {scheme} model with paper's configuration")

        # Paper: random start index i \in [0, NCB] to avoid assuming a fixed BS
        # beam; fixed start (i=0) only when experiments explicitly fix it. C3
        # uses random start, C1/C2 fixed per Config.VARIANTS.
        variant = config.VARIANTS.get(scheme, {"random_start": False, "start_index": 0})

        model = BeamAlignmentModel(
            num_tx_antennas=config.NTX,
            num_rx_antennas=config.NRX,
            codebook_size=config.NCB,
            num_sensing_steps=config.T,
            rnn_hidden_size=config.RNN_HIDDEN_SIZE,
            rnn_type=config.RNN_TYPE,
            num_feedback=config.NUM_FEEDBACK,
            start_beam_index=variant.get("start_index", 0) or 0,
            random_start=variant.get("random_start", False),
            scheme=scheme  # Pass scheme (C1, C2, or C3)
        )
        
        # CRITICAL: Mimic train.py exactly to ensure correct weight restoration
        # 1. Create Optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=1000,
            decay_rate=config.LEARNING_RATE_DECAY,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # 2. Initialize Optimizer Variables (Dummy Step)
        # This is required because the checkpoint contains optimizer variables.
        # If we don't initialize them, restore() might fail to map model weights correctly.
        print(f"  Initializing optimizer variables for {scheme}...")
        with tf.GradientTape() as tape:
            # Run with training=True to ensure all training-only variables are created
            dummy_results = model(batch_size=16, snr_db=config.SNR_TRAIN, training=True)
            # Dummy loss to get gradients
            dummy_loss = -tf.reduce_mean(dummy_results['beamforming_gain'])
        
        gradients = tape.gradient(dummy_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 3. Load Checkpoint (Structure must match train.py: model + optimizer)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_subdir, max_to_keep=1
        )
        
        if ckpt_manager.latest_checkpoint:
            # We can still use expect_partial() for safety, but it should match perfectly now
            status = checkpoint.restore(ckpt_manager.latest_checkpoint)
            
            # Verify that model variables were actually restored
            try:
                status.assert_existing_objects_matched()
                print(f"✓ Successfully loaded {scheme} from {ckpt_manager.latest_checkpoint}")
            except AssertionError as e:
                print(f"⚠ WARNING: Checkpoint match error for {scheme}: {e}")
                # If exact match fails (e.g. optimizer var mismatch), try partial
                print("  Attempting partial restore...")
                status.expect_partial()
                print(f"✓ Loaded {scheme} (partial match)")
        else:
            print(f"⚠ No {scheme} checkpoint found in {checkpoint_subdir}, using untrained model")
        
        return model
    
    # Load models
    print("\nLoading C1 (Fixed codebook, Fixed start)...")
    model_c1 = load_scheme_model('C1', './checkpoints_C1_T16')
    
    print("\nLoading C2 (Learnable codebook, Fixed start)...")
    model_c2 = load_scheme_model('C2', './checkpoints_C2_T16')
    
    print("\nLoading C3 (Learnable codebook, Random start)...")
    model_c3 = load_scheme_model('C3', './checkpoints_C3_T16')
    
    # Evaluate C1
    print("\nEvaluating C1...")
    for snr_db in tqdm(snr_range, desc="C1"):
        metrics = evaluate_at_snr(model_c1, float(snr_db), num_samples, batch_size, target_snr_db)
        c1_results['bf_gain'].append(metrics['mean_bf_gain_db'])
        c1_results['sat_prob'].append(metrics['satisfaction_prob'])
    
    # Evaluate C2
    print("\nEvaluating C2...")
    for snr_db in tqdm(snr_range, desc="C2"):
        metrics = evaluate_at_snr(model_c2, float(snr_db), num_samples, batch_size, target_snr_db)
        c2_results['bf_gain'].append(metrics['mean_bf_gain_db'])
        c2_results['sat_prob'].append(metrics['satisfaction_prob'])
    
    # Evaluate C3
    print("\nEvaluating C3...")
    for snr_db in tqdm(snr_range, desc="C3"):
        metrics = evaluate_at_snr(model_c3, float(snr_db), num_samples, batch_size, target_snr_db)
        c3_results['bf_gain'].append(metrics['mean_bf_gain_db'])
        c3_results['sat_prob'].append(metrics['satisfaction_prob'])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: BF Gain vs SNR
    ax1.plot(snr_range, c1_results['bf_gain'], 'o-', 
             label='C1 (Fixed CB, Fixed Start)', linewidth=2, markersize=8, color='C0')
    ax1.plot(snr_range, c2_results['bf_gain'], 's-', 
             label='C2 (Learnable CB, Fixed Start)', linewidth=2, markersize=8, color='C1')
    ax1.plot(snr_range, c3_results['bf_gain'], '^-', 
             label='C3 (Learnable CB, Random Start)', linewidth=2.5, markersize=9, color='C2')
    
    ax1.set_xlabel('SNR [dB]', fontsize=14)
    ax1.set_ylabel('Beamforming gain [dB]', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='best')
    ax1.set_title('(a) Beamforming Gain vs SNR', fontsize=14)
    
    # Subplot 2: Satisfaction Probability vs SNR
    ax2.plot(snr_range, c1_results['sat_prob'], 'o-', 
             label='C1 (Fixed CB, Fixed Start)', linewidth=2, markersize=8, color='C0')
    ax2.plot(snr_range, c2_results['sat_prob'], 's-', 
             label='C2 (Learnable CB, Fixed Start)', linewidth=2, markersize=8, color='C1')
    ax2.plot(snr_range, c3_results['sat_prob'], '^-', 
             label='C3 (Learnable CB, Random Start)', linewidth=2.5, markersize=9, color='C2')
    
    ax2.set_xlabel('SNR [dB]', fontsize=14)
    ax2.set_ylabel('Satisfaction probability', fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc='best')
    ax2.set_title('(b) Satisfaction Probability vs SNR', fontsize=14)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure_4_c1_c2_c3.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Figure 4 to {fig_path}")
    plt.close()
    
    return c1_results, c2_results, c3_results


def generate_figure_5_c2_c3(config, output_dir='./results', num_samples=1000):
    """
    Figure 5: Performance vs number of sensing steps T (C2 vs C3)
    
    This function properly implements Figure 5 from the paper, which compares
    C2 and C3 schemes across different T values. Each T value requires a
    separately trained model.
    
    From the paper: "Another aspect we want to examine is the impact of T, i.e., 
    the number of sensing steps, on the performance for a fixed NCB...To also 
    investigate the influence of the learnable codebook on the performance 
    (especially if T≤NCB), the experiment is performed for both the variants C2 and C3."
    
    Args:
        config: Configuration object
        output_dir: Output directory for plots
        num_samples: Number of samples for evaluation
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 5: Performance vs Sensing Steps T (C2 vs C3)")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # T values from paper
    T_values = np.array([1, 3, 5, 7, 8, 9, 15])
    batch_size = 256
    snr_db = 5.0  # SNR_ANT = 5 dB from paper 
    target_snr_db = 20.0
    
    # Storage
    c2_results = {'bf_gain': [], 'sat_prob': []}
    c3_results = {'bf_gain': [], 'sat_prob': []}
    
    # Helper function to load model for specific T
    def load_model_for_T(scheme, T_val):
        """Load a model trained with specific T value."""
        
        print(f"\n  Creating {scheme} model with T={T_val}")
        
        model = BeamAlignmentModel(
            num_tx_antennas=config.NTX,
            num_rx_antennas=config.NRX,
            codebook_size=config.NCB,
            num_sensing_steps=T_val,  # Use specific T value
            rnn_hidden_size=config.RNN_HIDDEN_SIZE,
            rnn_type=config.RNN_TYPE,
            num_feedback=config.NUM_FEEDBACK,
            start_beam_index=0,
            random_start=(scheme == 'C3'),  # C2: fixed start, C3: random start 
            scheme=scheme
        )
        
        # Create Optimizer (required for checkpoint loading)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=1000,
            decay_rate=config.LEARNING_RATE_DECAY,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Initialize Optimizer Variables (Dummy Step)
        print(f"  Initializing optimizer variables for {scheme} T={T_val}...")
        with tf.GradientTape() as tape:
            dummy_results = model(batch_size=16, snr_db=config.SNR_TRAIN, training=True)
            dummy_loss = -tf.reduce_mean(dummy_results['beamforming_gain'])
        
        gradients = tape.gradient(dummy_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Load Checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_dir = f'./checkpoints_{scheme}_T{T_val}'
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, ckpt_dir, max_to_keep=1
        )
        
        if ckpt_manager.latest_checkpoint:
            status = checkpoint.restore(ckpt_manager.latest_checkpoint)
            try:
                status.assert_existing_objects_matched()
                print(f"  ✓ Loaded {scheme} T={T_val} from {ckpt_manager.latest_checkpoint}")
            except AssertionError as e:
                print(f"  ⚠ WARNING: Checkpoint match error for {scheme} T={T_val}: {e}")
                print("    Attempting partial restore...")
                status.expect_partial()
                print(f"  ✓ Loaded {scheme} T={T_val} (partial match)")
        else:
            print(f"  ⚠ No checkpoint found in {ckpt_dir}, using untrained model")
            print(f"    To train: python train.py --scheme {scheme} --num_sensing_steps {T_val}")
        
        return model
    
    # Evaluate C2 across all T values
    print("\nEvaluating C2 (Learnable codebook, Fixed start)...")
    for T in tqdm(T_values, desc="C2"):
        model_c2 = load_model_for_T('C2', int(T))
        metrics = evaluate_at_snr(model_c2, snr_db, num_samples, batch_size, target_snr_db)
        c2_results['bf_gain'].append(metrics['mean_bf_gain_db'])
        c2_results['sat_prob'].append(metrics['satisfaction_prob'])
    
    # Evaluate C3 across all T values
    print("\nEvaluating C3 (Learnable codebook, Random start)...")
    for T in tqdm(T_values, desc="C3"):
        model_c3 = load_model_for_T('C3', int(T))
        metrics = evaluate_at_snr(model_c3, snr_db, num_samples, batch_size, target_snr_db)
        c3_results['bf_gain'].append(metrics['mean_bf_gain_db'])
        c3_results['sat_prob'].append(metrics['satisfaction_prob'])
    
    # Print summary
    print("\n" + "=" * 80)
    print("FIGURE 5 SUMMARY")
    print("=" * 80)
    print(f"{'T':<5} {'C2 BF (dB)':<12} {'C2 Sat':<10} {'C3 BF (dB)':<12} {'C3 Sat':<10}")
    print("-" * 80)
    for i, T in enumerate(T_values):
        print(f"{T:<5} {c2_results['bf_gain'][i]:<12.2f} {c2_results['sat_prob'][i]:<10.3f} "
              f"{c3_results['bf_gain'][i]:<12.2f} {c3_results['sat_prob'][i]:<10.3f}")
    print("=" * 80)
    
    # Plot with dual y-axis (matching paper's Figure 5 style)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    # BF gain (left axis)
    color_c2 = 'C1'
    color_c3 = 'C2'
    ax1.plot(T_values, c2_results['bf_gain'], 's-', label='C2 (Fixed start)', 
             linewidth=2, markersize=8, color=color_c2)
    ax1.plot(T_values, c3_results['bf_gain'], '^-', label='C3 (Random start)', 
             linewidth=2.5, markersize=9, color=color_c3)
    ax1.set_xlabel('Number of sensing steps T', fontsize=14)
    ax1.set_ylabel('Beamforming gain [dB]', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Satisfaction probability (right axis)
    ax2 = ax1.twinx()
    ax2.plot(T_values, c2_results['sat_prob'], 's--', linewidth=2, markersize=8, 
             color=color_c2, alpha=0.6)
    ax2.plot(T_values, c3_results['sat_prob'], '^--', linewidth=2.5, markersize=9, 
             color=color_c3, alpha=0.6)
    ax2.set_ylabel('Satisfaction probability', fontsize=14, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim([0, 1.05])
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=12, loc='lower right')
    
    plt.title('Performance vs Sensing Steps (SNR_ANT = 5 dB)', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figure_5_c2_c3_performance_vs_T.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Figure 5 to {fig_path}")
    plt.close()
    
    return c2_results, c3_results


# ==================== Main Execution ====================


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Reproduce figures 4 and 5 from the paper (C1/C2/C3 comparisons)'
    )
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Output directory for plots')
    parser.add_argument('--figure', type=str, default='all', 
                       choices=['all', '4', '5'],
                       help='Which figure(s) to generate: 4 (C1/C2/C3 vs SNR), 5 (C2/C3 vs T)')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    setup_device(verbose=True)
    
    # Print config
    print("\n" + "=" * 80)
    print("PAPER FIGURE REPRODUCTION")
    print("=" * 80)
    print("\nGenerating correct figure implementations:")
    print("  - Figure 4: C1, C2, C3 comparison vs SNR")
    print("  - Figure 5: C2, C3 comparison vs T (sensing steps)")
    print("=" * 80)
    Config.print_config()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate figures
    if args.figure in ['all', '4']:
        generate_figure_4_c1_c2_c3(Config, args.output_dir, args.num_samples)
    
    if args.figure in ['all', '5']:
        generate_figure_5_c2_c3(Config, args.output_dir, args.num_samples)
    
    print("\n" + "=" * 80)
    print("FIGURE REPRODUCTION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated figures:")
    if args.figure in ['all', '4']:
        print("  - figure_4_c1_c2_c3.png")
    if args.figure in ['all', '5']:
        print("  - figure_5_c2_c3_performance_vs_T.png")


if __name__ == "__main__":
    main()
