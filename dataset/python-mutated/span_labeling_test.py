"""Tests for span_labeling network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.networks import span_labeling

@keras_parameterized.run_all_keras_modes
class SpanLabelingTest(keras_parameterized.TestCase):

    def test_network_creation(self):
        if False:
            while True:
                i = 10
        'Validate that the Keras object can be created.'
        sequence_length = 15
        input_width = 512
        test_network = span_labeling.SpanLabeling(input_width=input_width, output='predictions')
        sequence_data = tf.keras.Input(shape=(sequence_length, input_width), dtype=tf.float32)
        (start_outputs, end_outputs) = test_network(sequence_data)
        expected_output_shape = [None, sequence_length]
        self.assertEqual(expected_output_shape, start_outputs.shape.as_list())
        self.assertEqual(expected_output_shape, end_outputs.shape.as_list())

    def test_network_invocation(self):
        if False:
            while True:
                i = 10
        'Validate that the Keras object can be invoked.'
        sequence_length = 15
        input_width = 512
        test_network = span_labeling.SpanLabeling(input_width=input_width)
        sequence_data = tf.keras.Input(shape=(sequence_length, input_width), dtype=tf.float32)
        outputs = test_network(sequence_data)
        model = tf.keras.Model(sequence_data, outputs)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, sequence_length, input_width))
        (start_outputs, end_outputs) = model.predict(input_data)
        expected_output_shape = (batch_size, sequence_length)
        self.assertEqual(expected_output_shape, start_outputs.shape)
        self.assertEqual(expected_output_shape, end_outputs.shape)

    def test_network_invocation_with_internal_logit_output(self):
        if False:
            print('Hello World!')
        'Validate that the logit outputs are correct.'
        sequence_length = 15
        input_width = 512
        test_network = span_labeling.SpanLabeling(input_width=input_width, output='predictions')
        sequence_data = tf.keras.Input(shape=(sequence_length, input_width), dtype=tf.float32)
        output = test_network(sequence_data)
        model = tf.keras.Model(sequence_data, output)
        logit_model = tf.keras.Model(test_network.inputs, [test_network.start_logits, test_network.end_logits])
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, sequence_length, input_width))
        (start_outputs, end_outputs) = model.predict(input_data)
        (start_logits, end_logits) = logit_model.predict(input_data)
        expected_output_shape = (batch_size, sequence_length)
        self.assertEqual(expected_output_shape, start_outputs.shape)
        self.assertEqual(expected_output_shape, end_outputs.shape)
        self.assertEqual(expected_output_shape, start_logits.shape)
        self.assertEqual(expected_output_shape, end_logits.shape)
        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)
        start_softmax = softmax_model.predict(start_logits)
        self.assertAllClose(start_outputs, start_softmax)
        end_softmax = softmax_model.predict(end_logits)
        self.assertAllClose(end_outputs, end_softmax)

    def test_network_invocation_with_external_logit_output(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate that the logit outputs are correct.'
        sequence_length = 15
        input_width = 512
        test_network = span_labeling.SpanLabeling(input_width=input_width, output='predictions')
        logit_network = span_labeling.SpanLabeling(input_width=input_width, output='logits')
        logit_network.set_weights(test_network.get_weights())
        sequence_data = tf.keras.Input(shape=(sequence_length, input_width), dtype=tf.float32)
        output = test_network(sequence_data)
        logit_output = logit_network(sequence_data)
        model = tf.keras.Model(sequence_data, output)
        logit_model = tf.keras.Model(sequence_data, logit_output)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, sequence_length, input_width))
        (start_outputs, end_outputs) = model.predict(input_data)
        (start_logits, end_logits) = logit_model.predict(input_data)
        expected_output_shape = (batch_size, sequence_length)
        self.assertEqual(expected_output_shape, start_outputs.shape)
        self.assertEqual(expected_output_shape, end_outputs.shape)
        self.assertEqual(expected_output_shape, start_logits.shape)
        self.assertEqual(expected_output_shape, end_logits.shape)
        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)
        start_softmax = softmax_model.predict(start_logits)
        self.assertAllClose(start_outputs, start_softmax)
        end_softmax = softmax_model.predict(end_logits)
        self.assertAllClose(end_outputs, end_softmax)

    def test_serialize_deserialize(self):
        if False:
            print('Hello World!')
        network = span_labeling.SpanLabeling(input_width=128, activation='relu', initializer='zeros', output='predictions')
        new_network = span_labeling.SpanLabeling.from_config(network.get_config())
        _ = new_network.to_json()
        self.assertAllEqual(network.get_config(), new_network.get_config())

    def test_unknown_output_type_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
            _ = span_labeling.SpanLabeling(input_width=10, output='bad')
if __name__ == '__main__':
    tf.test.main()