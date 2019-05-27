import tensorflow as tf
from descrete_optimization import Capacitor, Projectile, Solenoid, Stage
from typing import List


@tf.function
def optimize_stage(efficiency: tf.Tensor, trainable_var: List[tf.Variable], optimizer: tf.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(-efficiency, trainable_var)
        optimizer.apply_gradients(zip(gradients, trainable_var))


if __name__ == "__main__":
    num_layers: tf.Variable = tf.Variable(2, name="num_layers", dtype=tf.int32)
    coil_width: tf.Variable = tf.Variable(0.1, name="coil_width", dtype=tf.float32)
    solenoid = Solenoid(num_layers, coil_width)
    capacitor = Capacitor(15, 650 * 10 ** -6)
    projectile = Projectile(0.015, 0.005)
    stage: Stage = Stage(solenoid, capacitor, projectile, 0.001)

    optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam()
    efficiency: tf.Tensor = stage.calculate_efficiency(0.01, 1000)
    optimize_stage(efficiency, [num_layers, coil_width], optimizer)
