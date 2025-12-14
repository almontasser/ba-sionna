"""
Checkpoint compatibility utilities.

Why this exists:
  - Older checkpoints in this repo were saved with `tf.train.Checkpoint(model=..., optimizer=...)`.
    Depending on TF/Keras version, some model weights (notably stacked RNN cells) may not have
    been serialized under the `model/` subtree and instead appeared under optimizer internals.
  - When the model architecture changes (e.g., different UE input features or RNN size),
    restoring those old checkpoints can crash with shape-mismatch errors.

This module provides lightweight, shape-based compatibility checks so training/evaluation can:
  - restore compatible checkpoints reliably, or
  - fail/skip restore with a clear message when incompatible.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Dict, List, Tuple

import tensorflow as tf


def _checkpoint_model_signature(model: tf.keras.Model) -> Dict[str, Tuple[int, ...]]:
    """
    Build the set of checkpoint keys+shapes that a *current* model expects.

    We do this by saving the model to a temporary checkpoint and reading
    `tf.train.list_variables()` from it. This is more robust than trying to map
    Keras variable `.path` strings to TF checkpoint keys (which can diverge
    across TF/Keras versions, especially for Dense kernels and nested models).
    """
    with tempfile.TemporaryDirectory(prefix="ckpt_signature_") as tmpdir:
        ckpt = tf.train.Checkpoint(model=model)
        path = ckpt.save(os.path.join(tmpdir, "ckpt"))
        vars_list = tf.train.list_variables(path)

    expected: Dict[str, Tuple[int, ...]] = {}
    for name, shape in vars_list:
        if name.startswith("model/"):
            expected[name] = tuple(int(d) for d in shape)
    return expected


@dataclass(frozen=True)
class CheckpointCompatibility:
    ok: bool
    missing: Tuple[str, ...]
    shape_mismatch: Tuple[Tuple[str, Tuple[int, ...], Tuple[int, ...]], ...]
    restored_fraction: float
    has_ue_rnn_kernel: bool

    def summary(self, *, max_items: int = 8) -> str:
        lines: List[str] = []
        if self.ok:
            lines.append("compatible")
        else:
            lines.append("incompatible")

        lines.append(f"restored_fraction≈{self.restored_fraction:.3f}")
        lines.append(f"has_ue_rnn_kernel={self.has_ue_rnn_kernel}")

        if self.shape_mismatch:
            lines.append("shape_mismatch:")
            for key, ckpt_shape, model_shape in list(self.shape_mismatch)[:max_items]:
                lines.append(f"  - {key}: ckpt{ckpt_shape} != model{model_shape}")
            if len(self.shape_mismatch) > max_items:
                lines.append(f"  ... (+{len(self.shape_mismatch) - max_items} more)")

        if self.missing:
            lines.append("missing:")
            for key in list(self.missing)[:max_items]:
                lines.append(f"  - {key}")
            if len(self.missing) > max_items:
                lines.append(f"  ... (+{len(self.missing) - max_items} more)")

        return "\n".join(lines)


def check_checkpoint_compatibility(
    model: tf.keras.Model,
    checkpoint_path: str,
    *,
    require_all_trainable: bool = True,
    require_ue_rnn_kernel: bool = True,
) -> CheckpointCompatibility:
    """
    Compare a checkpoint against the current model *by variable key + shape*.

    This does not attempt to restore. It is meant to prevent hard crashes due to
    shape mismatch when restoring checkpoints saved from a different architecture.
    """
    # Signature keys/shapes that *this* model expects under the `model/` subtree.
    expected = _checkpoint_model_signature(model)
    if not expected:
        # If this ever happens, something is fundamentally wrong with serialization.
        return CheckpointCompatibility(
            ok=False,
            missing=tuple(),
            shape_mismatch=tuple(),
            restored_fraction=0.0,
            has_ue_rnn_kernel=False,
        )

    ckpt_vars = tf.train.list_variables(checkpoint_path)
    ckpt_shapes: Dict[str, Tuple[int, ...]] = {
        name: tuple(int(d) for d in shape) for name, shape in ckpt_vars
    }

    missing: List[str] = []
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    present = 0
    for key, expected_shape in expected.items():
        if key not in ckpt_shapes:
            missing.append(key)
            continue
        present += 1
        ckpt_shape = ckpt_shapes[key]
        if ckpt_shape != expected_shape:
            shape_mismatch.append((key, ckpt_shape, expected_shape))

    restored_fraction = present / max(1, len(expected))

    has_ue_rnn_kernel = (
        "model/ue_controller/ue_gru_cell_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        in ckpt_shapes
        or "model/ue_controller/ue_lstm_cell_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        in ckpt_shapes
    )

    ok = not shape_mismatch
    if require_all_trainable:
        ok = ok and (not missing)
    if require_ue_rnn_kernel:
        ok = ok and has_ue_rnn_kernel

    return CheckpointCompatibility(
        ok=bool(ok),
        missing=tuple(missing),
        shape_mismatch=tuple(shape_mismatch),
        restored_fraction=float(restored_fraction),
        has_ue_rnn_kernel=bool(has_ue_rnn_kernel),
    )
