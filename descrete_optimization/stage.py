from descrete_optimization import Solenoid
from descrete_optimization import Capacitor
from descrete_optimization import Projectile
from typing import Tuple, List
import tensorflow as tf

IntTensor = tf.Tensor
FloatTensor = tf.Tensor
FloatTimeSeriesTensor = tf.Tensor  # Dtype tf.float, shape [num_time_steps]


class Stage:
    def __init__(self, solenoid: Solenoid, capacitor: Capacitor,
                 projectile: Projectile):
        self._solenoid = solenoid
        self._capacitor = capacitor
        self._projectile = projectile

    @tf.function
    def calculate_efficiency(self, dt: float,
                             num_intervals: int) -> FloatTensor:
        pass

    @tf.function
    def _damping_factor_op(self) -> FloatTensor:
        return (self._capacitor.esr + self._solenoid.resistance) / (
                2 * self._solenoid.inductance)

    @tf.function
    def _fundamental_frequency_op(self) -> FloatTensor:
        return 1 / tf.sqrt(
            self._solenoid.inductance * self._capacitor.capacitance)

    @tf.function
    def _b_op(self, fundamental_frequency: FloatTensor, inductance: FloatTensor,
              init_cap_voltage: float) -> Tuple[FloatTensor, FloatTensor]:
        return (tf.constant(0, dtype=tf.float32),
                init_cap_voltage / (fundamental_frequency * inductance))

    @tf.function
    def _discretized_flywheel_current_curve(self, time_range: List[int],
                                            fundamental_frequency: FloatTensor,
                                            dampening_factor: FloatTensor) \
            -> FloatTimeSeriesTensor:
        Icyc: FloatTimeSeriesTensor = self._capacitor.discretized_current_curve(
            time_range,
            self._solenoid.inductance,
            fundamental_frequency,
            dampening_factor)

        Idis: FloatTimeSeriesTensor = tf.math.reduce_max(Icyc) * tf.exp(
            (-self._solenoid.resistance + time_range) / (
                    self._solenoid.inductance * 10 ** -6))
        local_max_idx: IntTensor = tf.argmax(Icyc, axis=0)

        return tf.concat([Icyc[:local_max_idx], Idis[local_max_idx:]], axis=0)
