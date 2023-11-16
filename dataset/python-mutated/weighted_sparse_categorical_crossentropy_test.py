"""Tests for masked LM loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling import networks
from official.nlp.modeling.losses import weighted_sparse_categorical_crossentropy

@keras_parameterized.run_all_keras_modes
class ClassificationLossTest(keras_parameterized.TestCase):

    def create_lm_model(self, vocab_size, sequence_length, hidden_size, num_predictions, output='predictions'):
        if False:
            while True:
                i = 10
        xformer_stack = networks.TransformerEncoder(vocab_size=vocab_size, num_layers=1, sequence_length=sequence_length, hidden_size=hidden_size, num_attention_heads=4)
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        (lm_outputs, _) = xformer_stack([word_ids, mask, type_ids])
        test_network = networks.MaskedLM(num_predictions=num_predictions, input_width=lm_outputs.shape[-1], source_network=xformer_stack, output=output)
        lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
        masked_lm_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
        output = test_network([lm_input_tensor, masked_lm_positions])
        return tf.keras.Model([lm_input_tensor, masked_lm_positions], output)

    def create_classification_model(self, input_width, num_classes):
        if False:
            print('Hello World!')
        test_object = networks.Classification(input_width=input_width, num_classes=num_classes)
        pooled_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
        output = test_object(pooled_data)
        return tf.keras.Model(pooled_data, output)

    def test_per_example_loss_3d_input(self):
        if False:
            return 10
        'Test per-example loss with a 3-dimensional input, from a masked LM.'
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        model = self.create_lm_model(vocab_size=vocab_size, sequence_length=sequence_length, hidden_size=hidden_size, num_predictions=num_predictions)
        batch_size = 3
        lm_input_data = 10 * np.random.random_sample((batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(2, size=(batch_size, num_predictions))
        output_data = model.predict([lm_input_data, masked_position_data])
        labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels)
        expected_shape = [batch_size, num_predictions]
        self.assertEqual(expected_shape, per_example_loss_data.shape.as_list())
        self.assertNotAllClose(tf.zeros_like(per_example_loss_data), per_example_loss_data)

    def test_per_example_loss_2d_input(self):
        if False:
            print('Hello World!')
        'Test per-example loss with a 2-d input, from a classifier.'
        input_width = 512
        num_classes = 10
        model = self.create_classification_model(input_width, num_classes)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        output_data = model.predict(input_data)
        labels = np.random.randint(num_classes, size=batch_size)
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels)
        self.assertEqual([batch_size], per_example_loss_data.shape.as_list())
        self.assertNotAllClose(tf.zeros_like(per_example_loss_data), per_example_loss_data)

    def test_per_example_loss_weights_3d_input(self):
        if False:
            i = 10
            return i + 15
        'Test weighted per-example loss with a 3-d input, from a masked LM.'
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        model = self.create_lm_model(vocab_size=vocab_size, sequence_length=sequence_length, hidden_size=hidden_size, num_predictions=num_predictions)
        batch_size = 3
        lm_input_data = 10 * np.random.random_sample((batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(2, size=(batch_size, num_predictions))
        output_data = model.predict([lm_input_data, masked_position_data])
        labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
        weights = np.random.randint(2, size=(batch_size, num_predictions))
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels, weights=weights)
        expected_weighted_loss = per_example_loss_data * weights
        self.assertAllClose(expected_weighted_loss, per_example_loss_data)

    def test_per_example_loss_weights_2d_input(self):
        if False:
            while True:
                i = 10
        'Test weighted per-example loss with a 2-d input, from a classifier.'
        input_width = 512
        num_classes = 10
        model = self.create_classification_model(input_width, num_classes)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        output_data = model.predict(input_data)
        labels = np.random.randint(num_classes, size=batch_size)
        weights = np.random.randint(2, size=batch_size)
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels, weights=weights)
        expected_weighted_loss = per_example_loss_data * weights
        self.assertAllClose(expected_weighted_loss, per_example_loss_data)

    def test_loss_3d_input(self):
        if False:
            i = 10
            return i + 15
        'Test overall loss with a 3-dimensional input, from a masked LM.'
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        model = self.create_lm_model(vocab_size=vocab_size, sequence_length=sequence_length, hidden_size=hidden_size, num_predictions=num_predictions)
        batch_size = 3
        lm_input_data = 10 * np.random.random_sample((batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(2, size=(batch_size, num_predictions))
        output_data = model.predict([lm_input_data, masked_position_data])
        labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
        weights = np.random.randint(2, size=(batch_size, num_predictions))
        per_example_loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=weights)
        expected_shape = []
        self.assertEqual(expected_shape, per_example_loss_data.shape.as_list())
        self.assertNotAllClose(tf.zeros_like(per_example_loss_data), per_example_loss_data)

    def test_loss_2d_input(self):
        if False:
            i = 10
            return i + 15
        'Test overall loss with a 2-d input, from a classifier.'
        input_width = 512
        num_classes = 10
        model = self.create_classification_model(input_width, num_classes)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        output_data = model.predict(input_data)
        labels = np.random.randint(num_classes, size=batch_size)
        loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels)
        self.assertNotAllClose(0, loss_data)

    def test_loss_weights_3d_input(self):
        if False:
            for i in range(10):
                print('nop')
        'Test masked loss with a 3-dimensional input, from a masked LM.'
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        model = self.create_lm_model(vocab_size=vocab_size, sequence_length=sequence_length, hidden_size=hidden_size, num_predictions=num_predictions)
        batch_size = 3
        lm_input_data = 10 * np.random.random_sample((batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(2, size=(batch_size, num_predictions))
        output_data = model.predict([lm_input_data, masked_position_data])
        labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
        null_weights = np.zeros((batch_size, num_predictions))
        weighted_loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=null_weights)
        self.assertAllClose(0, weighted_loss_data)

    def test_loss_weights_2d_input(self):
        if False:
            while True:
                i = 10
        'Test masked loss with a 2-d input, from a classifier.'
        input_width = 512
        num_classes = 10
        model = self.create_classification_model(input_width, num_classes)
        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_width))
        output_data = model.predict(input_data)
        labels = np.random.randint(num_classes, size=batch_size)
        null_weights = np.zeros(batch_size)
        weighted_loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=null_weights)
        self.assertAllClose(0, weighted_loss_data)

    def test_mismatched_predictions_and_labels_ranks_squeezes(self):
        if False:
            while True:
                i = 10
        'Test that the loss asserts when rank(predictions)-1 != rank(labels).'
        batch_size = 3
        output_data = np.random.random_sample((batch_size, 10))
        labels = np.random.randint(10, size=(batch_size, 1))
        _ = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels)

    def test_mismatched_weights_and_labels_ranks_fail(self):
        if False:
            while True:
                i = 10
        'Test that the loss asserts when rank(predictions) != rank(labels).'
        batch_size = 3
        output_data = np.random.random_sample((batch_size, 10, 15))
        labels = np.random.randint(10, size=(batch_size, 10))
        weights = np.random.randint(2, size=batch_size)
        with self.assertRaisesRegex(RuntimeError, '.*of the same rank.*'):
            _ = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels, weights=weights)
        with self.assertRaisesRegex(RuntimeError, '.*of the same rank.*'):
            _ = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=weights)

    def test_tf_tensor_inputs(self):
        if False:
            while True:
                i = 10
        'Test that tf.Tensors can be used as inputs to the loss function.'
        batch_size = 3
        output_data = tf.convert_to_tensor(np.random.random_sample((batch_size, 10, 15)))
        labels = tf.convert_to_tensor(np.random.randint(10, size=(batch_size, 10)))
        weights = tf.convert_to_tensor(np.random.randint(2, size=(batch_size, 10)))
        _ = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels, weights=weights)
        _ = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=weights)

    def test_legacy_lm_loss_compatibility(self):
        if False:
            while True:
                i = 10
        'Test to validate computational correctness during refactors.'
        output_data = np.array([[[-2.5286622, -1.0963473, -1.4925185, -2.4451098, -1.2923571], [-2.7117882, -1.1205841, -4.02187, -0.9966936, -1.5119683]], [[-2.5379114, -0.82479054, -2.287932, -1.3747153, -2.053741], [-2.5379114, -0.82479054, -2.287932, -1.3747153, -2.053741]], [[-2.7760355, -1.8219438, -3.0924666, -1.0779881, -0.9407509], [-2.7760355, -1.8219438, -3.0924666, -1.0779881, -0.9407509]]])
        labels = np.array([[4, 0], [2, 2], [2, 1]])
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels)
        expected_per_example_loss_data = [[1.2923571, 2.7117882], [2.287932, 2.287932], [3.0924666, 1.8219438]]
        self.assertAllClose(expected_per_example_loss_data, per_example_loss_data)
        weights = np.array([[1, 0], [0, 0], [0, 0]])
        loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=weights)
        expected_loss_data = 1.2923441
        self.assertAllClose(expected_loss_data, loss_data)

    def test_legacy_classification_loss_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        'Test to validate computational correctness during refactors.'
        output_data = np.array([[-0.0016094601, -10.966038, -6.4434357], [-0.0016975292, -6.4009643, -10.226612]])
        labels = np.array([2, 1])
        per_example_loss_data = weighted_sparse_categorical_crossentropy.per_example_loss(predictions=output_data, labels=labels)
        expected_per_example_loss_data = [6.4434357, 6.4009643]
        self.assertAllClose(expected_per_example_loss_data, per_example_loss_data)
        weights = None
        loss_data = weighted_sparse_categorical_crossentropy.loss(predictions=output_data, labels=labels, weights=weights)
        expected_loss_data = 6.4222
        self.assertAllClose(expected_loss_data, loss_data)
if __name__ == '__main__':
    tf.test.main()