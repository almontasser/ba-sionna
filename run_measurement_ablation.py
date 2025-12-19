"""
Quick measurement-ablation runner to check whether the learned policy actually
depends on the sensing measurements y_t.

We evaluate on fresh channel realizations each batch, then apply:
  - none: normal operation
  - zero: y_t := 0
  - noise_only: y_t := w_t^H n_t
  - shuffle: y_t permuted across batch (break H↔y association)

If performance is similar across ablations, the model is effectively ignoring y_t.
"""

import argparse
from config import Config
from figures_evaluators.common import (
    load_c3_model,
    evaluate_at_snr_with_ablation,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    ap.add_argument("--scenario", type=str, default="UMi", help="One of UMi/UMa/RMa")
    ap.add_argument("--snr_db", type=float, default=5.0)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--target_snr_db", type=float, default=20.0)
    args = ap.parse_args()

    checkpoint_dir = args.checkpoint_dir or f"./checkpoints_C3_T{Config.T}"
    model = load_c3_model(Config, checkpoint_dir, scenarios=[args.scenario])

    for ablation in ["none", "zero", "noise_only", "shuffle"]:
        r = evaluate_at_snr_with_ablation(
            model,
            args.snr_db,
            args.num_samples,
            args.batch_size,
            args.target_snr_db,
            measurement_ablation=ablation,
        )
        print(
            f"{ablation:>9} | mean_gain_dB={r['mean_bf_gain_db']:.3f} "
            f"| sat={r['satisfaction_prob']:.3f}"
        )


if __name__ == "__main__":
    main()
