import unittest
from descrete_optimization import Capacitor, Projectile, Solenoid, Stage
import tensorflow as tf


class TestStage(unittest.TestCase):

    def test_stage_efficiency(self):
        num_layers: tf.Variable = tf.Variable(3, dtype=tf.int32)
        coil_width: tf.Variable = tf.Variable(0.02, dtype=tf.float32)
        solenoid = Solenoid(num_layers, coil_width)
        capacitor = Capacitor(15, 650)
        projectile = Projectile(15, 5, 5)
        stage: Stage = Stage(solenoid, capacitor, projectile, 0.001)
        print(stage.calculate_efficiency(0.01, 1000))


if __name__ == "__main__":
    unittest.main()
