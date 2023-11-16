"""Tests for BERT trainer network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from official.nlp.modeling import networks
from official.nlp.modeling.networks import bert_span_labeler

@keras_parameterized.run_all_keras_modes
class BertSpanLabelerTest(keras_parameterized.TestCase):

    def test_bert_trainer(self):
        if False:
            return 10
        'Validate that the Keras object can be created.'
        vocab_size = 100
        sequence_length = 512
        test_network = networks.TransformerEncoder(vocab_size=vocab_size, num_layers=2, sequence_length=sequence_length)
        bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        cls_outs = bert_trainer_model([word_ids, mask, type_ids])
        self.assertEqual(2, len(cls_outs))
        expected_shape = [None, sequence_length]
        for out in cls_outs:
            self.assertAllEqual(expected_shape, out.shape.as_list())

    def test_bert_trainer_named_compilation(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate compilation using explicit output names.'
        vocab_size = 100
        sequence_length = 512
        test_network = networks.TransformerEncoder(vocab_size=vocab_size, num_layers=2, sequence_length=sequence_length)
        bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)
        bert_trainer_model.compile(optimizer='sgd', loss={'start_positions': 'mse', 'end_positions': 'mse'})

    def test_bert_trainer_tensor_call(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate that the Keras object can be invoked.'
        test_network = networks.TransformerEncoder(vocab_size=100, num_layers=2, sequence_length=2)
        bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)
        word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
        type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        _ = bert_trainer_model([word_ids, mask, type_ids])

    def test_serialize_deserialize(self):
        if False:
            return 10
        'Validate that the BERT trainer can be serialized and deserialized.'
        test_network = networks.TransformerEncoder(vocab_size=100, num_layers=2, sequence_length=5)
        bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)
        config = bert_trainer_model.get_config()
        new_bert_trainer_model = bert_span_labeler.BertSpanLabeler.from_config(config)
        _ = new_bert_trainer_model.to_json()
        self.assertAllEqual(bert_trainer_model.get_config(), new_bert_trainer_model.get_config())
if __name__ == '__main__':
    tf.test.main()