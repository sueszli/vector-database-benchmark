"""Tests for the customized Swish activation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.modeling import activations

@keras_parameterized.run_all_keras_modes
class CustomizedSwishTest(keras_parameterized.TestCase):

    def _hard_swish_np(self, x):
        if False:
            print('Hello World!')
        x = np.float32(x)
        return x * np.clip(x + 3, 0, 6) / 6

    def test_simple_swish(self):
        if False:
            return 10
        features = [[0.25, 0, -0.25], [-1, -2, 3]]
        customized_swish_data = activations.simple_swish(features)
        swish_data = tf.nn.swish(features)
        self.assertAllClose(customized_swish_data, swish_data)

    def test_hard_swish(self):
        if False:
            return 10
        features = [[0.25, 0, -0.25], [-1, -2, 3]]
        customized_swish_data = activations.hard_swish(features)
        swish_data = self._hard_swish_np(features)
        self.assertAllClose(customized_swish_data, swish_data)
if __name__ == '__main__':
    tf.test.main()