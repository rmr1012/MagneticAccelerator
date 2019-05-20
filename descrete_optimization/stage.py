from descrete_optimization import Solenoid
from descrete_optimization import Capacitor
from descrete_optimization import Projectile

import tensorflow as tf
import math

IntTensor = tf.Tensor
FloatTensor = tf.Tensor
FloatTimeSeriesTensor = tf.Tensor  # Dtype tf.float, shape [num_time_steps]


class Stage:
    def __init__(self, solenoid: Solenoid, capacitor: Capacitor,
                 projectile: Projectile, offset: float,
                 space_permeability: float = 4 * math.pi * 10 ** -7):

        self._solenoid = solenoid
        self._capacitor = capacitor
        self._projectile = projectile
        self._coil_offset = offset  # meters
        self._space_permeability: float = space_permeability

    @tf.function
    def calculate_efficiency(self, duration: float,
                             num_steps: int) -> FloatTensor:
        dt: float = duration / num_steps
        time_steps: FloatTimeSeriesTensor = tf.linspace(0, duration, num_steps)
        flywheel_current: FloatTimeSeriesTensor = \
            self._discretized_flywheel_current_curve(time_steps)
        vel: FloatTensor = tf.constant(0, dtype=tf.float32)
        pos: FloatTensor = tf.constant(0, dtype=tf.float32)  # meters
        for i in range(1, num_steps):
            if pos < self._coil_offset:  # Before entering he coil.
                force: FloatTensor = \
                    ((self._solenoid.num_turns * flywheel_current[i]) ** 2
                     * self._space_permeability
                     * self._projectile.cross_sectional_area) \
                    / (2 * (self._coil_offset - pos) ** 2)
            elif pos - self._coil_offset <= self._solenoid.coil_width:
                # In coil.
                force: FloatTensor = \
                    (.5 - ((pos - self._coil_offset)
                           / self._solenoid.coil_width)) * 2 \
                    * (self._projectile.relative_permeability
                       * self._space_permeability * self._solenoid.num_turns * flywheel_current[i]) ** 2 \
                    * self._projectile.cross_sectional_area / (2 * self._space_permeability)
            else:
                force: FloatTensor = \
                    -(self._solenoid.num_turns * flywheel_current[i]) ** 2 * self._space_permeability \
                    * self._projectile.cross_sectional_area / (2 * (
                            pos - self._coil_offset - self._solenoid.coil_width)) ** 2

            force: FloatTensor = tf.abs(tf.minimum(force, self._projectile.max_force))
            acc: FloatTensor = force / self._projectile.mass
            vel: FloatTensor = vel + acc * dt
            pos: FloatTensor = pos + vel * dt

        return 0.5 * (self._projectile.mass) * vel ** 2 / self._capacitor.energy

    @tf.function
    def _discretized_flywheel_current_curve(
            self, time_range: FloatTimeSeriesTensor) -> FloatTimeSeriesTensor:
        dampening_factor: FloatTensor = \
            (self._capacitor.esr + self._solenoid.resistance) / (2 * self._solenoid.inductance)

        fundamental_frequency: FloatTensor = 1 / tf.sqrt(self._solenoid.inductance * self._capacitor.capacitance)

        Icyc: FloatTimeSeriesTensor = self._capacitor.discretized_current_curve(
            time_range,
            self._solenoid.inductance,
            fundamental_frequency,
            dampening_factor)

        Idis: FloatTimeSeriesTensor = tf.math.reduce_max(Icyc) * tf.exp(
            (-self._solenoid.resistance + time_range) / self._solenoid.inductance)
        local_max_idx: IntTensor = tf.argmax(Icyc, axis=0)

        return tf.concat([Icyc[:local_max_idx], Idis[local_max_idx:]], axis=0)
