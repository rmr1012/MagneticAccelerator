import unittest
import numpy as np
import descrete_optimization.solenoid as coil
import tensorflow as tf


class TestSolenoid(unittest.TestCase):

    def test_ctor_correct_permeability_resistance(self):
        num_layers: tf.Variable = tf.Variable(2, dtype=tf.int32)
        coil_width: tf.Variable = tf.Variable(20, dtype=tf.float32)
        solenoid: coil.Solenoid = coil.Solenoid(num_layers, coil_width)
        self.assertTrue(np.allclose(solenoid.inductance.numpy(),
                                    [76.67112310383358]))
        self.assertTrue(np.allclose(solenoid.resistance.numpy(),
                                    [0.4369790616788424]))


if __name__ == "__main__":
    unittest.main()
