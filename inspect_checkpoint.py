#!/usr/bin/env python3
"""
Checkpoint inspection / compatibility helper.

This repo changes model architecture fairly often (e.g., UE input features,
LayerNorm, mobility/time-varying channels). Old checkpoints can therefore fail
to restore with shape-mismatch errors.

This script prints:
  - the latest checkpoint path in a directory
  - a short compatibility summary vs the *current* model definition
  - (optional) checkpoint variable listing
"""

import argparse

import tensorflow as tf

from checkpoint_utils import check_checkpoint_compatibility
from config import Config
from train import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a checkpoint directory")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=f"{Config.CHECKPOINT_DIR}_C3_T{Config.T}",
        help="Directory containing ckpt-* files",
    )
    parser.add_argument(
        "--list_vars",
        action="store_true",
        help="List variables stored in the checkpoint",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=10,
        help="Max missing/mismatch items printed in the compatibility summary",
    )
    args = parser.parse_args()

    ckpt_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    if not ckpt_path:
        raise FileNotFoundError(f"No checkpoint found in '{args.checkpoint_dir}'.")

    print(f"Checkpoint: {ckpt_path}")

    model = create_model(Config)
    _ = model(batch_size=2, snr_db=Config.SNR_TRAIN, training=False)

    compat = check_checkpoint_compatibility(
        model,
        ckpt_path,
        require_all_trainable=True,
        require_ue_rnn_kernel=True,
    )
    print("\nCompatibility:")
    print(compat.summary(max_items=args.max_items))

    if args.list_vars:
        print("\nCheckpoint variables:")
        for name, shape in tf.train.list_variables(ckpt_path):
            print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()

