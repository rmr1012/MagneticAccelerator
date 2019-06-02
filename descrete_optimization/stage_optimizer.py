import tensorflow as tf
from typing import NamedTuple
from descrete_optimization import Capacitor, Projectile, Solenoid, Stage


class OptimizationParams(NamedTuple):
    num_layers: tf.Tensor
    coil_width: tf.Tensor
    inner_dia: tf.Tensor
    gauge: tf.Tensor
    stage: Stage


def create_stage() -> OptimizationParams:
    num_layers: tf.Tensor = tf.random.uniform([], minval=1, maxval=10, dtype=tf.int32)
    coil_width: tf.Tensor = tf.random.uniform([], minval=0.01, maxval=0.5, dtype=tf.float32)
    inner_dia: tf.Tensor = tf.constant(0.005, dtype=tf.float32)
    gauge: tf.Tensor = tf.random.uniform([], minval=1, maxval=41, dtype=tf.int32)
    solenoid = Solenoid(num_layers, coil_width, inner_dia, gauge)
    capacitor = Capacitor(15, 650 * 10 ** -6)
    projectile = Projectile(0.015, 0.005)
    stage: Stage = Stage(solenoid, capacitor, projectile, 0.001)
    return OptimizationParams(num_layers, coil_width, inner_dia, gauge, stage)


if __name__ == "__main__":
    opt_params: OptimizationParams = create_stage()
    print([opt_params.stage.calculate_efficiency(0.25, 2500).result() for i in range(20)])
