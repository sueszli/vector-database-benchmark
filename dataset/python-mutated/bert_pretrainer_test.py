"""Tests for BERT trainer network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling import networks
from official.nlp.modeling.networks import bert_pretrainer

@keras_parameterized.run_all_keras_modes
class BertPretrainerTest(keras_parameterized.TestCase):

    def test_bert_trainer(self):
        if False:
            print('Hello World!')
        'Validate that the Keras object can be created.'
        vocab_size = 100
        sequence_length = 512
        test_network = networks.TransformerEncoder(vocab_size=vocab_size, num_layers=2, sequence_length=sequence_length)
        num_classes = 3
        num_token_predictions = 2
        bert_trainer_model = bert_pretrainer.BertPretrainer(test_network, num_classes=num_classes, num_token_predictions=num_token_predictions)
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        lm_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        (lm_outs, cls_outs) = bert_trainer_model([word_ids, mask, type_ids, lm_mask])
        expected_lm_shape = [None, num_token_predictions, vocab_size]
        expected_classification_shape = [None, num_classes]
        self.assertAllEqual(expected_lm_shape, lm_outs.shape.as_list())
        self.assertAllEqual(expected_classification_shape, cls_outs.shape.as_list())

    def test_bert_trainer_tensor_call(self):
        if False:
            print('Hello World!')
        'Validate that the Keras object can be invoked.'
        test_network = networks.TransformerEncoder(vocab_size=100, num_layers=2, sequence_length=2)
        bert_trainer_model = bert_pretrainer.BertPretrainer(test_network, num_classes=2, num_token_predictions=2)
        word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
        type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        lm_mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
        (_, _) = bert_trainer_model([word_ids, mask, type_ids, lm_mask])

    def test_serialize_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate that the BERT trainer can be serialized and deserialized.'
        test_network = networks.TransformerEncoder(vocab_size=100, num_layers=2, sequence_length=5)
        bert_trainer_model = bert_pretrainer.BertPretrainer(test_network, num_classes=4, num_token_predictions=3)
        config = bert_trainer_model.get_config()
        new_bert_trainer_model = bert_pretrainer.BertPretrainer.from_config(config)
        _ = new_bert_trainer_model.to_json()
        self.assertAllEqual(bert_trainer_model.get_config(), new_bert_trainer_model.get_config())
if __name__ == '__main__':
    tf.test.main()