import tensorflow as tf

IntTensorVar = tf.Variable
FloatTensorVar = tf.Variable
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


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

    def get_voltage(self, t: IntTensor) -> float:
        return self.initial_voltage

    def get_energy(self, t: IntTensor) -> float:
        return 0.5 * (self.capacitance * 10 ** -6) * self.get_voltage(t) ** 2
