import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from evaluate_paper_figures import evaluate_at_snr
from models.beam_alignment import BeamAlignmentModel


def generate_figure_5_c2_c3(config, output_dir="./results", num_samples=1000):
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
    c2_results = {"bf_gain": [], "sat_prob": []}
    c3_results = {"bf_gain": [], "sat_prob": []}

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
            random_start=(scheme == "C3"),  # C2: fixed start, C3: random start
            scheme=scheme,
        )

        # Create Optimizer (required for checkpoint loading)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=1000,
            decay_rate=config.LEARNING_RATE_DECAY,
            staircase=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Initialize Optimizer Variables (Dummy Step)
        print(f"  Initializing optimizer variables for {scheme} T={T_val}...")
        with tf.GradientTape() as tape:
            dummy_results = model(batch_size=16, snr_db=config.SNR_TRAIN, training=True)
            dummy_loss = -tf.reduce_mean(dummy_results["beamforming_gain"])

        gradients = tape.gradient(dummy_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Load Checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_dir = f"./checkpoints_{scheme}_T{T_val}"
        ckpt_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=1)

        if ckpt_manager.latest_checkpoint:
            status = checkpoint.restore(ckpt_manager.latest_checkpoint)
            try:
                status.assert_existing_objects_matched()
                print(
                    f"  ✓ Loaded {scheme} T={T_val} from {ckpt_manager.latest_checkpoint}"
                )
            except AssertionError as e:
                print(
                    f"  ⚠ WARNING: Checkpoint match error for {scheme} T={T_val}: {e}"
                )
                print("    Attempting partial restore...")
                status.expect_partial()
                print(f"  ✓ Loaded {scheme} T={T_val} (partial match)")
        else:
            print(f"  ⚠ No checkpoint found in {ckpt_dir}, using untrained model")
            print(
                f"    To train: python train.py --scheme {scheme} --num_sensing_steps {T_val}"
            )

        return model

    # Evaluate C2 across all T values
    print("\nEvaluating C2 (Learnable codebook, Fixed start)...")
    for T in tqdm(T_values, desc="C2"):
        model_c2 = load_model_for_T("C2", int(T))
        metrics = evaluate_at_snr(
            model_c2, snr_db, num_samples, batch_size, target_snr_db
        )
        c2_results["bf_gain"].append(metrics["mean_bf_gain_db"])
        c2_results["sat_prob"].append(metrics["satisfaction_prob"])

    # Evaluate C3 across all T values
    print("\nEvaluating C3 (Learnable codebook, Random start)...")
    for T in tqdm(T_values, desc="C3"):
        model_c3 = load_model_for_T("C3", int(T))
        metrics = evaluate_at_snr(
            model_c3, snr_db, num_samples, batch_size, target_snr_db
        )
        c3_results["bf_gain"].append(metrics["mean_bf_gain_db"])
        c3_results["sat_prob"].append(metrics["satisfaction_prob"])

    # Print summary
    print("\n" + "=" * 80)
    print("FIGURE 5 SUMMARY")
    print("=" * 80)
    print(
        f"{'T':<5} {'C2 BF (dB)':<12} {'C2 Sat':<10} {'C3 BF (dB)':<12} {'C3 Sat':<10}"
    )
    print("-" * 80)
    for i, T in enumerate(T_values):
        print(
            f"{T:<5} {c2_results['bf_gain'][i]:<12.2f} {c2_results['sat_prob'][i]:<10.3f} "
            f"{c3_results['bf_gain'][i]:<12.2f} {c3_results['sat_prob'][i]:<10.3f}"
        )
    print("=" * 80)

    # Plot with dual y-axis (matching paper's Figure 5 style)
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # BF gain (left axis)
    color_c2 = "C1"
    color_c3 = "C2"
    ax1.plot(
        T_values,
        c2_results["bf_gain"],
        "s-",
        label="C2 (Fixed start)",
        linewidth=2,
        markersize=8,
        color=color_c2,
    )
    ax1.plot(
        T_values,
        c3_results["bf_gain"],
        "^-",
        label="C3 (Random start)",
        linewidth=2.5,
        markersize=9,
        color=color_c3,
    )
    ax1.set_xlabel("Number of sensing steps T", fontsize=14)
    ax1.set_ylabel("Beamforming gain [dB]", fontsize=14, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)

    # Satisfaction probability (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
        T_values,
        c2_results["sat_prob"],
        "s--",
        linewidth=2,
        markersize=8,
        color=color_c2,
        alpha=0.6,
    )
    ax2.plot(
        T_values,
        c3_results["sat_prob"],
        "^--",
        linewidth=2.5,
        markersize=9,
        color=color_c3,
        alpha=0.6,
    )
    ax2.set_ylabel("Satisfaction probability", fontsize=14, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim([0, 1.05])

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=12, loc="lower right")

    plt.title("Performance vs Sensing Steps (SNR_ANT = 5 dB)", fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "figure_5_c2_c3_performance_vs_T.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Figure 5 to {fig_path}")
    plt.close()

    return c2_results, c3_results
