"""Tests for Keras-based einsum layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.layers import dense_einsum

@keras_parameterized.run_all_keras_modes
class DenseEinsumLayer(keras_parameterized.TestCase):

    def test_3D_einsum_with_two_bound_dimensions(self):
        if False:
            while True:
                i = 10
        test_layer = dense_einsum.DenseEinsum(output_shape=(64,), num_summed_dimensions=2)
        input_tensor = tf.keras.Input(shape=(None, 40, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(test_layer._einsum_string, 'abcd,cde->abe')
        self.assertEqual(test_layer._kernel_shape, (40, 80, 64))

    def test_3D_einsum_with_one_bound_dimensions(self):
        if False:
            while True:
                i = 10
        test_layer = dense_einsum.DenseEinsum(output_shape=(64, 32), num_summed_dimensions=1)
        input_tensor = tf.keras.Input(shape=(None, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(test_layer._einsum_string, 'abc,cde->abde')
        self.assertEqual(test_layer._kernel_shape, (80, 64, 32))

    def test_2D_einsum_with_one_bound_dimensions(self):
        if False:
            i = 10
            return i + 15
        test_layer = dense_einsum.DenseEinsum(output_shape=(64,), num_summed_dimensions=1)
        input_tensor = tf.keras.Input(shape=(None, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(test_layer._einsum_string, 'abc,cd->abd')
        self.assertEqual(test_layer._kernel_shape, (80, 64))

    def test_bias_term_can_be_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        test_layer = dense_einsum.DenseEinsum(output_shape=64, num_summed_dimensions=1, use_bias=True)
        input_tensor = tf.keras.Input(shape=(None, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(2, len(test_layer.get_weights()))
        test_layer = dense_einsum.DenseEinsum(output_shape=64, num_summed_dimensions=1, use_bias=False)
        input_tensor = tf.keras.Input(shape=(None, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(1, len(test_layer.get_weights()))

    def test_activation(self):
        if False:
            for i in range(10):
                print('nop')
        no_activation_layer = dense_einsum.DenseEinsum(output_shape=64, num_summed_dimensions=1, activation=None)
        input_tensor = tf.keras.Input(shape=(None, 80))
        output_tensor = no_activation_layer(input_tensor)
        no_activation_model = tf.keras.Model(input_tensor, output_tensor)
        activation_layer = dense_einsum.DenseEinsum(output_shape=64, num_summed_dimensions=1, activation='softmax')
        input_tensor = tf.keras.Input(shape=(None, 80))
        output_tensor = activation_layer(input_tensor)
        activation_model = tf.keras.Model(input_tensor, output_tensor)
        activation_model.set_weights(no_activation_model.get_weights())
        input_values = 10 * np.random.random_sample((10, 4, 80))
        non_activated_data = no_activation_model.predict(input_values)
        activated_data = activation_model.predict(input_values)
        self.assertNotAllClose(activated_data, non_activated_data)

    def test_non_iterable_output_shape(self):
        if False:
            while True:
                i = 10
        test_layer = dense_einsum.DenseEinsum(output_shape=64, num_summed_dimensions=1)
        input_tensor = tf.keras.Input(shape=(None, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(test_layer._einsum_string, 'abc,cd->abd')
        self.assertEqual(test_layer._kernel_shape, (80, 64))

    def test_with_explicit_initializer(self):
        if False:
            i = 10
            return i + 15
        test_layer = dense_einsum.DenseEinsum(output_shape=(64,), num_summed_dimensions=2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        input_tensor = tf.keras.Input(shape=(None, 40, 80))
        _ = test_layer(input_tensor)
        self.assertEqual(test_layer._einsum_string, 'abcd,cde->abe')
        self.assertEqual(test_layer._kernel_shape, (40, 80, 64))
if __name__ == '__main__':
    tf.test.main()