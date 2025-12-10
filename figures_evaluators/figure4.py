import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from evaluate_paper_figures import evaluate_at_snr
from models.beam_alignment import BeamAlignmentModel


def generate_figure_4_c1_c2_c3(config, output_dir="./results", num_samples=2000):
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
    c1_results = {"bf_gain": [], "sat_prob": []}
    c2_results = {"bf_gain": [], "sat_prob": []}
    c3_results = {"bf_gain": [], "sat_prob": []}

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
            scheme=scheme,  # Pass scheme (C1, C2, or C3)
        )

        # CRITICAL: Mimic train.py exactly to ensure correct weight restoration
        # 1. Create Optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=1000,
            decay_rate=config.LEARNING_RATE_DECAY,
            staircase=True,
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
            dummy_loss = -tf.reduce_mean(dummy_results["beamforming_gain"])

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
                print(
                    f"✓ Successfully loaded {scheme} from {ckpt_manager.latest_checkpoint}"
                )
            except AssertionError as e:
                print(f"⚠ WARNING: Checkpoint match error for {scheme}: {e}")
                # If exact match fails (e.g. optimizer var mismatch), try partial
                print("  Attempting partial restore...")
                status.expect_partial()
                print(f"✓ Loaded {scheme} (partial match)")
        else:
            print(
                f"⚠ No {scheme} checkpoint found in {checkpoint_subdir}, using untrained model"
            )

        return model

    # Load models
    print("\nLoading C1 (Fixed codebook, Fixed start)...")
    model_c1 = load_scheme_model("C1", "./checkpoints_C1_T16")

    print("\nLoading C2 (Learnable codebook, Fixed start)...")
    model_c2 = load_scheme_model("C2", "./checkpoints_C2_T16")

    print("\nLoading C3 (Learnable codebook, Random start)...")
    model_c3 = load_scheme_model("C3", "./checkpoints_C3_T16")

    # Evaluate C1
    print("\nEvaluating C1...")
    for snr_db in tqdm(snr_range, desc="C1"):
        metrics = evaluate_at_snr(
            model_c1, float(snr_db), num_samples, batch_size, target_snr_db
        )
        c1_results["bf_gain"].append(metrics["mean_bf_gain_db"])
        c1_results["sat_prob"].append(metrics["satisfaction_prob"])

    # Evaluate C2
    print("\nEvaluating C2...")
    for snr_db in tqdm(snr_range, desc="C2"):
        metrics = evaluate_at_snr(
            model_c2, float(snr_db), num_samples, batch_size, target_snr_db
        )
        c2_results["bf_gain"].append(metrics["mean_bf_gain_db"])
        c2_results["sat_prob"].append(metrics["satisfaction_prob"])

    # Evaluate C3
    print("\nEvaluating C3...")
    for snr_db in tqdm(snr_range, desc="C3"):
        metrics = evaluate_at_snr(
            model_c3, float(snr_db), num_samples, batch_size, target_snr_db
        )
        c3_results["bf_gain"].append(metrics["mean_bf_gain_db"])
        c3_results["sat_prob"].append(metrics["satisfaction_prob"])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: BF Gain vs SNR
    ax1.plot(
        snr_range,
        c1_results["bf_gain"],
        "o-",
        label="C1 (Fixed CB, Fixed Start)",
        linewidth=2,
        markersize=8,
        color="C0",
    )
    ax1.plot(
        snr_range,
        c2_results["bf_gain"],
        "s-",
        label="C2 (Learnable CB, Fixed Start)",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax1.plot(
        snr_range,
        c3_results["bf_gain"],
        "^-",
        label="C3 (Learnable CB, Random Start)",
        linewidth=2.5,
        markersize=9,
        color="C2",
    )

    ax1.set_xlabel("SNR [dB]", fontsize=14)
    ax1.set_ylabel("Beamforming gain [dB]", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc="best")
    ax1.set_title("(a) Beamforming Gain vs SNR", fontsize=14)

    # Subplot 2: Satisfaction Probability vs SNR
    ax2.plot(
        snr_range,
        c1_results["sat_prob"],
        "o-",
        label="C1 (Fixed CB, Fixed Start)",
        linewidth=2,
        markersize=8,
        color="C0",
    )
    ax2.plot(
        snr_range,
        c2_results["sat_prob"],
        "s-",
        label="C2 (Learnable CB, Fixed Start)",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax2.plot(
        snr_range,
        c3_results["sat_prob"],
        "^-",
        label="C3 (Learnable CB, Random Start)",
        linewidth=2.5,
        markersize=9,
        color="C2",
    )

    ax2.set_xlabel("SNR [dB]", fontsize=14)
    ax2.set_ylabel("Satisfaction probability", fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc="best")
    ax2.set_title("(b) Satisfaction Probability vs SNR", fontsize=14)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure_4_c1_c2_c3.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Figure 4 to {fig_path}")
    plt.close()

    return c1_results, c2_results, c3_results
