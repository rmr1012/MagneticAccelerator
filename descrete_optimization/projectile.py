import math
import tensorflow as tf

IntTensor = tf.Tensor
FloatTensor = tf.Tensor


class Projectile:
    def __init__(self,
                 mass: float,
                 diameter: float,
                 relative_permeability: float = 6.3 * 10 ** -3,
                 saturation: float = 0.75,
                 space_permeability: float = 4 * math.pi * 10 ** -7):
        """
        TODO: complete doc.
        :param mass:
        :param diameter:
        :param relative_permeability:
        :param saturation: measured in T (Tesla).
        :param space_permeability:
        """
        self.mass: float = mass
        self.relative_permeability: float = relative_permeability
        self.cross_sectional_area: float = math.pi * diameter
        self.max_force = saturation ** 2 * self.cross_sectional_area / (2 * space_permeability)
