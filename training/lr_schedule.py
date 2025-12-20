"""
Learning-rate schedules used for training.
"""

from __future__ import annotations

import tensorflow as tf


class ScaledSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Multiply a base schedule by a (mutable) scale factor.

    This lets the training loop "back off" the LR without recreating the optimizer.
    """

    def __init__(
        self,
        schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
        *,
        initial_scale: float = 1.0,
        name: str = "lr_scale",
    ):
        self.schedule = schedule
        self.scale = tf.Variable(float(initial_scale), trainable=False, dtype=tf.float32, name=name)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        return tf.cast(self.scale, tf.float32) * tf.cast(self.schedule(step), tf.float32)

    def set_scale(self, scale: float) -> None:
        self.scale.assign(float(scale))

    def multiply_scale(self, factor: float) -> None:
        self.scale.assign(self.scale * float(factor))


class ConstantLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Constant learning-rate schedule (useful for quick LR sweeps)."""

    def __init__(self, learning_rate: float):
        self.learning_rate = float(learning_rate)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        del step
        return tf.constant(self.learning_rate, dtype=tf.float32)
