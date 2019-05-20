import unittest
import descrete_optimization.capacitor as cap
import tensorflow as tf


class TestCapacitor(unittest.TestCase):

    def test_capacitor_current_curve(self):
        capacitor: cap.Capacitor = cap.Capacitor(15, 650, 0)
        self.assertAlmostEqual(capacitor.energy, 73125)


if __name__ == "__main__":
    unittest.main()
