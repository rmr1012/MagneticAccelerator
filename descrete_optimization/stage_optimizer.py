import tensorflow as tf
from typing import NamedTuple
from descrete_optimization import Capacitor, Projectile, Solenoid, Stage


class OptimizationParams(NamedTuple):
    num_layers: tf.Tensor
    coil_width: tf.Variable
    capacitance: tf.Variable
    inner_dia: tf.Tensor
    gauge: tf.Tensor
    stage: Stage
    offset: tf.Variable
    def __str__(self):
        return str(self.num_layers.numpy())+"lrs "+str(round(self.coil_width.numpy()*100,3))+" w(cm) "+str(round(self.capacitance.numpy()*10**6,3))+" uF "+str(self.gauge.numpy())+"AWG "+str(round(self.offset.numpy()*100,3))+" o(cm)"


def create_stage() -> OptimizationParams:
    num_layers: tf.Tensor = tf.random.uniform([], minval=1, maxval=10, dtype=tf.int32)
    coil_width: tf.Variable = tf.Variable(tf.random.uniform([], minval=0.01, maxval=0.08, dtype=tf.float32), # 1cm to 8cm coil width
                                          trainable=True)
    capacitance: tf.Variable = tf.Variable(tf.random.uniform([], minval=10 ** -4, maxval=10 ** -1, dtype=tf.float32), # 100uF - 100mF Cap
                                          trainable=True)
    offset: tf.Variable = tf.Variable(tf.random.uniform([], minval=0.001, maxval=0.01, dtype=tf.float32),
                                          trainable=True)
    inner_dia: tf.Tensor = tf.constant(0.005, dtype=tf.float32)
    gauge: tf.Tensor = tf.random.uniform([], minval=15, maxval=41, dtype=tf.int32) # gauge 15-41
    solenoid = Solenoid(num_layers, coil_width, inner_dia, gauge)
    capacitor = Capacitor(20, capacitance, esr=0.05) #20V
    projectile = Projectile(mass=0.0027, diameter=0.00476) #2.7g $ 4.76mm diameter
    stage: Stage = Stage(solenoid, capacitor, projectile, offset)
    return OptimizationParams(num_layers, coil_width, capacitance, inner_dia, gauge, stage, offset)


def train_step(optimizer, opt_params):
    with tf.GradientTape() as tape:
        loss = 1 - opt_params.stage.calculate_efficiency(0.25, 2500)
    gradients = tape.gradient(loss, [opt_params.coil_width])
    optimizer.apply_gradients(zip(gradients, [opt_params.coil_width]))
    return loss


if __name__ == "__main__":
    NUM_HPARAM_SEARCH = 10
    NUM_TRAIN_STEPS = 10
    for i in range(NUM_HPARAM_SEARCH):
        optimizer = tf.keras.optimizers.Adam(10.1)
        opt_params: OptimizationParams = create_stage()
        for i in range(NUM_TRAIN_STEPS):
            loss = train_step(optimizer, opt_params)
            print("step ", i, " - loss: ", loss.numpy(), str(opt_params))
