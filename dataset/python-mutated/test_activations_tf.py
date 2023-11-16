import unittest
import numpy as np
from transformers import is_tf_available
from transformers.testing_utils import require_tf
if is_tf_available():
    import tensorflow as tf
    from transformers.activations_tf import get_tf_activation

@require_tf
class TestTFActivations(unittest.TestCase):

    def test_gelu_10(self):
        if False:
            while True:
                i = 10
        x = tf.constant([-100, -1.0, -0.1, 0, 0.1, 1.0, 100.0])
        gelu = get_tf_activation('gelu')
        gelu10 = get_tf_activation('gelu_10')
        y_gelu = gelu(x)
        y_gelu_10 = gelu10(x)
        clipped_mask = tf.where(y_gelu_10 < 10.0, 1.0, 0.0)
        self.assertEqual(tf.math.reduce_max(y_gelu_10).numpy().item(), 10.0)
        self.assertTrue(np.allclose(y_gelu * clipped_mask, y_gelu_10 * clipped_mask))

    def test_get_activation(self):
        if False:
            return 10
        get_tf_activation('gelu')
        get_tf_activation('gelu_10')
        get_tf_activation('gelu_fast')
        get_tf_activation('gelu_new')
        get_tf_activation('glu')
        get_tf_activation('mish')
        get_tf_activation('quick_gelu')
        get_tf_activation('relu')
        get_tf_activation('sigmoid')
        get_tf_activation('silu')
        get_tf_activation('swish')
        get_tf_activation('tanh')
        with self.assertRaises(KeyError):
            get_tf_activation('bogus')
        with self.assertRaises(KeyError):
            get_tf_activation(None)