"""Contains the get_layer_policy function.

This is a separate file from policy.py to avoid a circular dependency.
get_layer_policy() relies on base_layer.py, itself which relies on policy.py.
"""
from tensorflow.python.keras.engine import base_layer

def get_layer_policy(layer):
    if False:
        for i in range(10):
            print('nop')
    'Returns the dtype policy of a layer.\n\n  Warning: This function is deprecated. Use\n  `tf.keras.layers.Layer.dtype_policy` instead.\n\n  Args:\n    layer: A `tf.keras.layers.Layer`.\n\n  Returns:\n    The `tf.keras.mixed_precision.Policy` of the layer.\n  '
    if not isinstance(layer, base_layer.Layer):
        raise ValueError('get_policy can only be called on a layer, but got: %s' % (layer,))
    return layer.dtype_policy