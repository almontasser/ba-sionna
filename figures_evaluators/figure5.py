import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from figures_evaluators.common import evaluate_at_snr, load_c3_model


def generate_figure_5_cdl_comparison(config, output_dir="./results", num_samples=1000):
    """
    Figure 5 (same axes as paper): performance vs number of sensing steps T.

    Change vs paper: instead of comparing C2 vs C3, we compare channel variants
    (CDL-A..E) using C3-only models trained for each T.
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 5: C3 PERFORMANCE VS T ACROSS CDL VARIANTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Keep the same T axis as the existing implementation
    T_values = np.array([1, 3, 5, 7, 8, 9, 15])
    batch_size = config.BATCH_SIZE
    snr_db = 5.0  # SNR_ANT = 5 dB (paper Figure 5)
    target_snr_db = float(getattr(config, "SNR_TARGET", 20.0))

    cdl_variants = list(getattr(config, "CDL_MODELS", ["A", "B", "C", "D", "E"]))
    results = {cdl: {"bf_gain": [], "sat_prob": []} for cdl in cdl_variants}

    for cdl in cdl_variants:
        print(f"\nEvaluating CDL-{cdl} across T...")
        for T in tqdm(T_values, desc=f"CDL-{cdl}"):
            ckpt_dir = f"./checkpoints_C3_T{int(T)}"
            model = load_c3_model(
                config,
                ckpt_dir,
                num_sensing_steps=int(T),
                cdl_models=[cdl],
            )
            metrics = evaluate_at_snr(
                model, snr_db, num_samples, batch_size, target_snr_db
            )
            results[cdl]["bf_gain"].append(metrics["mean_bf_gain_db"])
            results[cdl]["sat_prob"].append(metrics["satisfaction_prob"])

    # Plot with dual y-axis (same axes style as the existing implementation)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    # Use matplotlib's default color cycle
    for cdl in cdl_variants:
        line_bf, = ax1.plot(
            T_values,
            results[cdl]["bf_gain"],
            "-",
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=f"CDL-{cdl}",
        )
        ax2.plot(
            T_values,
            results[cdl]["sat_prob"],
            "--",
            linewidth=2.0,
            marker="o",
            markersize=6,
            color=line_bf.get_color(),
            alpha=0.6,
        )

    ax1.set_xlabel("Number of sensing steps T", fontsize=14)
    ax1.set_ylabel("Beamforming gain [dB]", fontsize=14, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Satisfaction probability", fontsize=14, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim([0, 1.05])

    ax1.legend(fontsize=12, loc="lower right")
    plt.title("Performance vs Sensing Steps (SNR_ANT = 5 dB)", fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "figure_5_cdl_comparison_vs_T.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Figure 5 to {fig_path}")
    plt.close()

    return results

