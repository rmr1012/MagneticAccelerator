import tensorflow as tf
from typing import List

IntTensorVar = tf.Variable
FloatTensorVar = tf.Variable
IntTensor = tf.Tensor
FloatTensor = tf.Tensor
FloatTimeSeriesTensor = tf.Tensor  # Dtype tf.float, shape [num_time_steps]


class Capacitor:

    def __init__(self, initial_voltage: FloatTensor, capacitance: float,
                 esr: float = 0):
        """
        TODO: complete doc.
        :param initial_voltage:
        :param capacitance:
        :param esr: equivalent series inductance.
        """

        self.esr: float = esr
        self.initial_voltage: float = initial_voltage
        self.capacitance: float = capacitance
        self.energy: float = 0.5 * (
                self.capacitance * 10 ** -6) * self.initial_voltage ** 2

    @tf.function
    def discretized_current_curve(self, time_range: List[int],
                                  inductance: FloatTensor,
                                  fundamental_frequency: FloatTensor,
                                  dampening_factor: FloatTensor) \
            -> FloatTimeSeriesTensor:
        constant: FloatTensor = self.initial_voltage / (
                inductance * 10 ** -6 * fundamental_frequency)
        return constant * tf.exp(-dampening_factor * time_range) * tf.sin(
            fundamental_frequency * time_range)
