"""Tests for transformer-based text encoder network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.networks import transformer_encoder

@keras_parameterized.run_all_keras_modes
class TransformerEncoderTest(keras_parameterized.TestCase):

    def test_network_creation(self):
        if False:
            while True:
                i = 10
        hidden_size = 32
        sequence_length = 21
        test_network = transformer_encoder.TransformerEncoder(vocab_size=100, hidden_size=hidden_size, sequence_length=sequence_length, num_attention_heads=2, num_layers=3)
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        (data, pooled) = test_network([word_ids, mask, type_ids])
        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())
        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    def test_network_creation_with_float16_dtype(self):
        if False:
            i = 10
            return i + 15
        hidden_size = 32
        sequence_length = 21
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        test_network = transformer_encoder.TransformerEncoder(vocab_size=100, hidden_size=hidden_size, sequence_length=sequence_length, num_attention_heads=2, num_layers=3, float_dtype='float16')
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        (data, pooled) = test_network([word_ids, mask, type_ids])
        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())
        self.assertAllEqual(tf.float16, data.dtype)
        self.assertAllEqual(tf.float16, pooled.dtype)

    def test_network_invocation(self):
        if False:
            for i in range(10):
                print('nop')
        hidden_size = 32
        sequence_length = 21
        vocab_size = 57
        num_types = 7
        tf.keras.mixed_precision.experimental.set_policy('float32')
        test_network = transformer_encoder.TransformerEncoder(vocab_size=vocab_size, hidden_size=hidden_size, sequence_length=sequence_length, num_attention_heads=2, num_layers=3, type_vocab_size=num_types)
        self.assertTrue(test_network._position_embedding_layer._use_dynamic_slicing)
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        (data, pooled) = test_network([word_ids, mask, type_ids])
        model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
        batch_size = 3
        word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
        mask_data = np.random.randint(2, size=(batch_size, sequence_length))
        type_id_data = np.random.randint(num_types, size=(batch_size, sequence_length))
        _ = model.predict([word_id_data, mask_data, type_id_data])
        max_sequence_length = 128
        test_network = transformer_encoder.TransformerEncoder(vocab_size=vocab_size, hidden_size=hidden_size, sequence_length=sequence_length, max_sequence_length=max_sequence_length, num_attention_heads=2, num_layers=3, type_vocab_size=num_types)
        self.assertTrue(test_network._position_embedding_layer._use_dynamic_slicing)
        model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
        _ = model.predict([word_id_data, mask_data, type_id_data])

    def test_serialize_deserialize(self):
        if False:
            return 10
        kwargs = dict(vocab_size=100, hidden_size=32, num_layers=3, num_attention_heads=2, sequence_length=21, max_sequence_length=21, type_vocab_size=12, intermediate_size=1223, activation='relu', dropout_rate=0.05, attention_dropout_rate=0.22, initializer='glorot_uniform', float_dtype='float16')
        network = transformer_encoder.TransformerEncoder(**kwargs)
        expected_config = dict(kwargs)
        expected_config['activation'] = tf.keras.activations.serialize(tf.keras.activations.get(expected_config['activation']))
        expected_config['initializer'] = tf.keras.initializers.serialize(tf.keras.initializers.get(expected_config['initializer']))
        self.assertEqual(network.get_config(), expected_config)
        new_network = transformer_encoder.TransformerEncoder.from_config(network.get_config())
        _ = new_network.to_json()
        self.assertAllEqual(network.get_config(), new_network.get_config())
if __name__ == '__main__':
    assert tf.version.VERSION.startswith('2.')
    tf.test.main()