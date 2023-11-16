"""Tests for Keras-based masked softmax layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.layers import masked_softmax

@keras_parameterized.run_all_keras_modes
class MaskedSoftmaxLayerTest(keras_parameterized.TestCase):

    def test_non_masked_softmax(self):
        if False:
            i = 10
            return i + 15
        test_layer = masked_softmax.MaskedSoftmax()
        input_tensor = tf.keras.Input(shape=(4, 8))
        output = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output)
        input_data = 10 * np.random.random_sample((3, 4, 8))
        output_data = model.predict(input_data)
        expected_data = tf.nn.softmax(input_data)
        self.assertAllClose(expected_data, output_data)

    def test_masked_softmax(self):
        if False:
            return 10
        test_layer = masked_softmax.MaskedSoftmax()
        input_tensor = tf.keras.Input(shape=(4, 8))
        mask_tensor = tf.keras.Input(shape=(4, 8))
        output = test_layer([input_tensor, mask_tensor])
        model = tf.keras.Model([input_tensor, mask_tensor], output)
        input_data = 10 * np.random.random_sample((3, 4, 8))
        mask_data = np.random.randint(2, size=(3, 4, 8))
        output_data = model.predict([input_data, mask_data])
        expected_zeros = np.greater(mask_data, 0)
        is_zeros = np.greater(output_data, 0)
        self.assertAllEqual(expected_zeros, is_zeros)

    def test_masked_softmax_with_none_mask(self):
        if False:
            while True:
                i = 10
        test_layer = masked_softmax.MaskedSoftmax()
        input_tensor = tf.keras.Input(shape=(4, 8))
        output = test_layer([input_tensor, None])
        model = tf.keras.Model(input_tensor, output)
        input_data = 10 * np.random.random_sample((3, 4, 8))
        output_data = model.predict(input_data)
        expected_data = tf.nn.softmax(input_data)
        self.assertAllClose(expected_data, output_data)

    def test_softmax_with_axes_expansion(self):
        if False:
            return 10
        test_layer = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
        input_tensor = tf.keras.Input(shape=(4, 8))
        mask_tensor = tf.keras.Input(shape=8)
        output = test_layer([input_tensor, mask_tensor])
        model = tf.keras.Model([input_tensor, mask_tensor], output)
        input_data = 10 * np.random.random_sample((3, 4, 8))
        mask_data = np.random.randint(2, size=(3, 8))
        output_data = model.predict([input_data, mask_data])
        expanded_mask = np.expand_dims(mask_data, axis=1) * np.ones_like(input_data)
        expected_zeros = np.greater(expanded_mask, 0)
        is_zeros = np.greater(output_data, 0)
        self.assertAllEqual(expected_zeros, is_zeros)
if __name__ == '__main__':
    tf.test.main()