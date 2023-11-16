"""Tests for Keras-based one-hot embedding layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.layers import on_device_embedding

@keras_parameterized.run_all_keras_modes
class OnDeviceEmbeddingTest(keras_parameterized.TestCase):

    def test_layer_creation(self):
        if False:
            print('Hello World!')
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [None, sequence_length, embedding_width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(output_tensor.dtype, tf.float32)

    def test_layer_creation_with_float16_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, dtype='float16')
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [None, sequence_length, embedding_width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(output_tensor.dtype, tf.float16)

    def test_layer_invocation(self):
        if False:
            return 10
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        batch_size = 3
        input_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
        output = model.predict(input_data)
        self.assertEqual(tf.float32, output.dtype)

    def test_layer_invocation_with_float16_dtype(self):
        if False:
            while True:
                i = 10
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, dtype='float16')
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        batch_size = 3
        input_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
        output = model.predict(input_data)
        self.assertEqual(tf.float16, output.dtype)

    def test_one_hot_layer_creation(self):
        if False:
            return 10
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, use_one_hot=True)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [None, sequence_length, embedding_width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(output_tensor.dtype, tf.float32)

    def test_one_hot_layer_creation_with_float16_dtype(self):
        if False:
            return 10
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, dtype='float16', use_one_hot=True)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        expected_output_shape = [None, sequence_length, embedding_width]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(output_tensor.dtype, tf.float16)

    def test_one_hot_layer_invocation(self):
        if False:
            print('Hello World!')
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, use_one_hot=True)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        batch_size = 3
        input_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
        output = model.predict(input_data)
        self.assertEqual(tf.float32, output.dtype)

    def test_one_hot_layer_invocation_with_float16_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_size = 31
        embedding_width = 27
        test_layer = on_device_embedding.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, dtype='float16', use_one_hot=True)
        sequence_length = 23
        input_tensor = tf.keras.Input(shape=sequence_length, dtype=tf.int32)
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        batch_size = 3
        input_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
        output = model.predict(input_data)
        self.assertEqual(tf.float16, output.dtype)
if __name__ == '__main__':
    tf.test.main()