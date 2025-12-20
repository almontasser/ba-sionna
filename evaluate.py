# ==================== Main Execution ====================


import argparse
import os
from config import Config
from device_setup import setup_device
from figures_evaluators.figure4 import generate_figure_4_scenario_comparison
from figures_evaluators.figure5 import generate_figure_5_scenario_comparison


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Generate Fig. 4/5 style plots comparing TR 38.901 scenarios (C3-only)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory for plots"
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["all", "4", "5"],
        help="Which figure(s) to generate: 4 (scenario vs SNR), 5 (scenario vs T)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help='Comma-separated scenarios to evaluate (e.g., "UMi,UMa,RMa"). Default: Config.SCENARIOS.',
    )
    parser.add_argument(
        "--checkpoint_dir_template",
        type=str,
        default=None,
        help=(
            "Checkpoint directory template with {T} and {scenario}, "
            'e.g. "./checkpoints_C3_T{T}_{scenario}".'
        ),
    )

    args = parser.parse_args()

    # Setup device
    setup_device(verbose=True)

    # Print config
    print("\n" + "=" * 80)
    print("PAPER FIGURE REPRODUCTION")
    print("=" * 80)
    print("\nGenerating figure-style comparisons (axes preserved):")
    print("  - Figure 4: Scenario comparison vs SNR (BF gain + satisfaction)")
    print("  - Figure 5: Scenario comparison vs T (BF gain + satisfaction)")
    print("=" * 80)
    Config.print_config()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.scenarios is not None:
        Config.SCENARIOS = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    # Generate figures
    if args.figure in ["all", "4"]:
        generate_figure_4_scenario_comparison(
            Config,
            args.output_dir,
            args.num_samples,
            checkpoint_dir_template=args.checkpoint_dir_template,
        )

    if args.figure in ["all", "5"]:
        generate_figure_5_scenario_comparison(
            Config,
            args.output_dir,
            args.num_samples,
            checkpoint_dir_template=args.checkpoint_dir_template,
        )

    print("\n" + "=" * 80)
    print("FIGURE REPRODUCTION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated figures:")
    if args.figure in ["all", "4"]:
        print("  - figure_4_scenario_comparison.png")
    if args.figure in ["all", "5"]:
        print("  - figure_5_scenario_comparison_vs_T.png")


if __name__ == "__main__":
    main()
