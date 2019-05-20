import unittest
from descrete_optimization import Capacitor
import tensorflow as tf
import numpy as np


class TestCapacitor(unittest.TestCase):

    def test_capacitor_current_curve(self):
        cap: Capacitor = Capacitor(15, 650, 0)
        self.assertAlmostEqual(cap.energy, 73125)
        self.assertTrue(np.allclose(
            cap.discretized_current_curve(tf.linspace(0.0, 0.01, 10),
                                                tf.constant(5, tf.float32),
                                                tf.constant(0.5, tf.float32),
                                                tf.constant(0.2, tf.float32)),
            [0., 0.00333259, 0.0066637, 0.00999333, 0.01332148, 0.01664814,
             0.01997331, 0.02329701, 0.02661921, 0.02993993]))


if __name__ == "__main__":
    unittest.main()
