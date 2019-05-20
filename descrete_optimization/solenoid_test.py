import unittest
from descrete_optimization import Solenoid
import tensorflow as tf


class TestSolenoid(unittest.TestCase):

    def test_ctor_correct_permeability_resistance(self):
        num_layers: tf.Variable = tf.Variable(2, dtype=tf.int32)
        coil_width: tf.Variable = tf.Variable(0.02, dtype=tf.float32)
        solenoid: Solenoid = Solenoid(num_layers, coil_width)
        print(solenoid.inductance.numpy())
        self.assertAlmostEqual(solenoid.inductance.numpy(), 0.00076671123103833)
        self.assertAlmostEqual(solenoid.resistance.numpy(), 0.4369790616788424)


if __name__ == "__main__":
    unittest.main()
