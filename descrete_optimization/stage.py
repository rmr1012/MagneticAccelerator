from typing import Tuple, List
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
        self._coil_offset = offset
        self._space_permeability: float = space_permeability

    @tf.function
    def calculate_efficiency(self, duration: float,
                             num_steps: int) -> FloatTensor:
        time_steps: FloatTimeSeriesTensor = tf.linspace(0, duration, num_steps)
        flywheel_current: FloatTimeSeriesTensor = \
            self._discretized_flywheel_current_curve(time_steps)
        acc: FloatTensor = tf.constant(0, dtype=tf.float32)
        vel: FloatTensor = tf.constant(0, dtype=tf.float32)
        x_prev: FloatTensor = tf.constant(0, dtype=tf.float32)
        x_curr: FloatTensor = x_prev
        for i in range(1, num_steps):
            force: FloatTensor = tf.constant(0, dtype=tf.float32)
            if x_prev < self._coil_offset:  # Before entering he coil.
                force: FloatTensor = \
                    ((self._solenoid.num_turns * flywheel_current[i]) ** 2
                     * self._space_permeability
                     * self._projectile.cross_sectional_area * 10 ** -6) \
                    / (2 * (self._coil_offset - x_prev) ** 2)
            elif x_prev - self._coil_offset <= self._solenoid.coil_width:
                # In coil.
                force: FloatTensor = \
                    (.5 - ((x_prev - self._coil_offset)
                           / self._solenoid.coil_width)) * 2 \
                    * (self._projectile.relative_permeability
                       * self._space_permeability * self._solenoid.num_turns
                       * flywheel_current[i]) ** 2 \
                    * self._projectile.cross_sectional_area * 10 ** -6 / (
                            2 * self._space_permeability)
            else:
                force: FloatTensor = \
                    -(self.coil.n * flywheel_current[i]) ** 2 * u0 \
                    * self.bullet.caliber * 10 ** -6 / 2 / ((
                                x_prev - self.offset * 10 ** -3 - self.coil.width * 10 ** -3)) ** 2

    @tf.function
    def _discretized_flywheel_current_curve(
            self, time_range: FloatTimeSeriesTensor) -> FloatTimeSeriesTensor:
        dampening_factor: FloatTensor = \
            (self._capacitor.esr + self._solenoid.resistance) / (
                    2 * self._solenoid.inductance)

        fundamental_frequency: FloatTensor = 1 / tf.sqrt(
            self._solenoid.inductance * self._capacitor.capacitance)

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
