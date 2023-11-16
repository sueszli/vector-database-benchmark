"""Tests for Keras-based positional embedding layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.layers import position_embedding

@keras_parameterized.run_all_keras_modes
class PositionEmbeddingLayerTest(keras_parameterized.TestCase):

    def test_static_layer_output_shape(self):
        if False:
            return 10
        test_layer = position_embedding.PositionEmbedding()
        sequence_length = 21
        width = 30
        input_tensor = tf.keras.Input(shape=(sequence_length, width))
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [1, sequence_length, width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(tf.float32, output_tensor.dtype)

    def test_float16_dtype(self):
        if False:
            i = 10
            return i + 15
        test_layer = position_embedding.PositionEmbedding(dtype='float16')
        sequence_length = 21
        width = 30
        input_tensor = tf.keras.Input(shape=(sequence_length, width))
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [1, sequence_length, width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(tf.float16, output_tensor.dtype)

    def test_dynamic_layer_output_shape(self):
        if False:
            print('Hello World!')
        max_sequence_length = 40
        test_layer = position_embedding.PositionEmbedding(use_dynamic_slicing=True, max_sequence_length=max_sequence_length)
        width = 30
        input_tensor = tf.keras.Input(shape=(None, width))
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [1, None, width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

    def test_dynamic_layer_slicing(self):
        if False:
            while True:
                i = 10
        max_sequence_length = 40
        test_layer = position_embedding.PositionEmbedding(use_dynamic_slicing=True, max_sequence_length=max_sequence_length)
        width = 30
        input_tensor = tf.keras.Input(shape=(None, width))
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        input_length = 17
        input_data = np.ones((1, input_length, width))
        output_data = model.predict(input_data)
        self.assertAllEqual([1, input_length, width], output_data.shape)
if __name__ == '__main__':
    tf.test.main()