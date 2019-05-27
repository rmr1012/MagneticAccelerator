import math
import tensorflow as tf

# Because python side loaded type hinting, this is the result.
IntTensorVar = tf.Variable
FloatTensorVar = tf.Variable
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


class Solenoid:
    wire_diameter: IntTensor = tf.constant(
        [7.3481, 6.5437, 5.8273, 5.1894,
         4.6213, 4.1154, 3.6649, 3.2636,
         2.9064, 2.5882, 2.3048, 2.0525,
         1.8278, 1.6277, 1.4495, 1.2908,
         1.1495, 1.0237, 0.9116, 0.8118,
         0.7229, 0.6438, 0.5733, 0.5106,
         0.4547, 0.4049, 0.3606, 0.3211,
         0.2859, 0.2546, 0.2268, 0.2019,
         0.1798, 0.1601, 0.1426, 0.1270,
         0.1131, 0.1007, 0.0897, 0.0799], dtype=tf.float32)

    gauge2dia_mm: tf.lookup.StaticHashTable = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.range(1, 41, dtype=tf.int32), wire_diameter), -1)

    def __init__(self,
                 num_layers: IntTensorVar,
                 coil_width: FloatTensorVar,
                 inner_dia: FloatTensorVar = tf.Variable(0.005, dtype=tf.float32),
                 gauge: IntTensorVar = tf.Variable(28, dtype=tf.int32),
                 space_permeability: float = 4 * math.pi * 10 ** -7,
                 copper_resistance: float = 1.68 * 10 ** -8):
        """
        TODO: complete doc.
        :param num_layers: Number of layers of wires for this solenoid.
        :param coil_width: The width of the solenoid in m.
        :param inner_dia: Inner diameter of the passage for the projectile.
        :param gauge: Wire gauge.
        :param space_permeability: Permeability of the space.
        """

        float_num_layers: FloatTensor = tf.dtypes.cast(num_layers, tf.float32)
        wire_dia: FloatTensor = self.gauge2dia_mm.lookup(gauge) * 10 ** -3
        turns_per_layer: IntTensor = tf.math.ceil(coil_width / wire_dia)
        ''' Unused properties in optimization but useful to know.
        outer_dia: FloatTensor = float_inner_dia + \
                                       float_num_layers * wire_dia * 2
        '''
        wire_length: IntTensor = sum([math.pi * calculate_diameter(layer, inner_dia, wire_dia)
                                      * turns_per_layer for layer in range(float_num_layers)])

        self.num_turns: FloatTensor = turns_per_layer * tf.dtypes.cast(num_layers, tf.float32)
        self.coil_width: FloatTensor = coil_width

        self.inductance: FloatTensor = sum(
            [calculate_layer_permiability(
                turns_per_layer,
                space_permeability,
                calculate_diameter(layer, inner_dia, wire_dia),
                wire_dia) for layer in range(float_num_layers)])  # Henry

        self.resistance: FloatTensor = copper_resistance * wire_length / (math.pi * (wire_dia / 2) ** 2)


@tf.function
def calculate_diameter(layer: int, inner_dia: FloatTensorVar, wire_dia: FloatTensor) -> FloatTensor:
    return inner_dia + 2 * layer * wire_dia


@tf.function
def calculate_layer_permiability(turns_per_layer: IntTensor,
                                 permeability: float,
                                 diameter: FloatTensor,
                                 wire_dia: FloatTensor) -> FloatTensor:
    return turns_per_layer ** 2 * permeability * (diameter / 2) * (tf.math.log(8 * diameter / wire_dia) - 2)
