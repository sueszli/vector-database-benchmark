"""Tests for classification network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling.networks import classification

@keras_parameterized.run_all_keras_modes
class ClassificationTest(keras_parameterized.TestCase):

    def test_network_creation(self):
        if False:
            i = 10
            return i + 15
        'Validate that the Keras object can be created.'
        input_width = 512
        num_classes = 10
        test_object = classification.Classification(input_width=input_width, num_classes=num_classes)
        cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(cls_data)
        expected_output_shape = [None, num_classes]
        self.assertEqual(expected_output_shape, output.shape.as_list())

    def test_network_invocation(self):
        if False:
            while True:
                i = 10
        'Validate that the Keras object can be invoked.'
        input_width = 512
        num_classes = 10
        test_object = classification.Classification(input_width=input_width, num_classes=num_classes, output='predictions')
        cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        input_data = 10 * np.random.random_sample((3, input_width))
        _ = model.predict(input_data)

    def test_network_invocation_with_internal_logits(self):
        if False:
            while True:
                i = 10
        'Validate that the logit outputs are correct.'
        input_width = 512
        num_classes = 10
        test_object = classification.Classification(input_width=input_width, num_classes=num_classes, output='predictions')
        cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(test_object.inputs, test_object.logits)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)
        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)
        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)
        calculated_softmax = softmax_model.predict(logits)
        self.assertAllClose(outputs, calculated_softmax)

    def test_network_invocation_with_internal_and_external_logits(self):
        if False:
            i = 10
            return i + 15
        'Validate that the logit outputs are correct.'
        input_width = 512
        num_classes = 10
        test_object = classification.Classification(input_width=input_width, num_classes=num_classes, output='logits')
        cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(test_object.inputs, test_object.logits)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)
        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)
        self.assertAllClose(outputs, logits)

    def test_network_invocation_with_logit_output(self):
        if False:
            return 10
        'Validate that the logit outputs are correct.'
        input_width = 512
        num_classes = 10
        test_object = classification.Classification(input_width=input_width, num_classes=num_classes, output='predictions')
        logit_object = classification.Classification(input_width=input_width, num_classes=num_classes, output='logits')
        logit_object.set_weights(test_object.get_weights())
        cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(cls_data)
        logit_output = logit_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(cls_data, logit_output)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)
        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)
        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)
        calculated_softmax = softmax_model.predict(logits)
        self.assertAllClose(outputs, calculated_softmax)

    def test_serialize_deserialize(self):
        if False:
            while True:
                i = 10
        network = classification.Classification(input_width=128, num_classes=10, initializer='zeros', output='predictions')
        new_network = classification.Classification.from_config(network.get_config())
        _ = new_network.to_json()
        self.assertAllEqual(network.get_config(), new_network.get_config())

    def test_unknown_output_type_fails(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
            _ = classification.Classification(input_width=128, num_classes=10, output='bad')
if __name__ == '__main__':
    tf.test.main()