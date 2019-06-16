import tensorflow as tf
from typing import NamedTuple
from descrete_optimization import Capacitor, Projectile, Solenoid, Stage


class OptimizationParams(NamedTuple):
    num_layers: tf.Tensor
    coil_width: tf.Variable
    capacitant: tf.Variable
    inner_dia: tf.Tensor
    gauge: tf.Tensor
    stage: Stage


def create_stage() -> OptimizationParams:
    num_layers: tf.Tensor = tf.random.uniform([], minval=1, maxval=10, dtype=tf.int32)
    coil_width: tf.Variable = tf.Variable(tf.random.uniform([], minval=0.01, maxval=0.5, dtype=tf.float32),
                                          trainable=True)
    capacitant: tf.Variable = tf.Variable(tf.random.uniform([], minval=10 ** -7, maxval=10 ** -4, dtype=tf.float32),
                                          trainable=True)
    inner_dia: tf.Tensor = tf.constant(0.005, dtype=tf.float32)
    gauge: tf.Tensor = tf.random.uniform([], minval=1, maxval=41, dtype=tf.int32)
    solenoid = Solenoid(num_layers, coil_width, inner_dia, gauge)
    capacitor = Capacitor(20, capacitant)
    projectile = Projectile(0.015, 0.005)
    stage: Stage = Stage(solenoid, capacitor, projectile, 0.001)
    return OptimizationParams(num_layers, coil_width, capacitant, inner_dia, gauge, stage)


def train_step(optimizer, opt_params):
    with tf.GradientTape() as tape:
        loss = 1 - opt_params.stage.calculate_efficiency(0.25, 2500)
    gradients = tape.gradient(loss, [opt_params.coil_width])
    optimizer.apply_gradients(zip(gradients, [opt_params.coil_width]))
    return loss


if __name__ == "__main__":
    NUM_HPARAM_SEARCH = 50
    NUM_TRAIN_STEPS = 20
    for i in range(NUM_HPARAM_SEARCH):
        optimizer = tf.keras.optimizers.Adam(10.1)
        opt_params: OptimizationParams = create_stage()
        for i in range(NUM_TRAIN_STEPS):
            loss = train_step(optimizer, opt_params)
            print("step ", i, " - loss: ", loss.numpy(), "\t coil_width: ", opt_params.coil_width.numpy())
