"""
Learning-rate schedules used for training.
"""

from __future__ import annotations

import tensorflow as tf


class WarmupThenSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up to `base_lr`, then delegate to an underlying schedule.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
    ):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.schedule = schedule

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.int64)
        if self.warmup_steps > 0:
            warm_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(
                self.warmup_steps, tf.float32
            )
            return tf.where(step < self.warmup_steps, warm_lr, self.schedule(step - self.warmup_steps))
        return self.schedule(step)


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


class WarmupThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up to `base_lr`, then exponential decay.
    """

    def __init__(self, base_lr: float, warmup_steps: int, decay_steps: int, decay_rate: float):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(base_lr),
            decay_steps=int(decay_steps),
            decay_rate=float(decay_rate),
            staircase=True,
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        if self.warmup_steps > 0:
            warm_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(
                self.warmup_steps, tf.float32
            )
            return tf.where(step < self.warmup_steps, warm_lr, self.decay(step - self.warmup_steps))
        return self.decay(step)
