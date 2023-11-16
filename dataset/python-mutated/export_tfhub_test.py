"""Tests official.nlp.bert.export_tfhub."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import bert_modeling
from official.nlp.bert import export_tfhub

class ExportTfhubTest(tf.test.TestCase):

    def test_export_tfhub(self):
        if False:
            i = 10
            return i + 15
        bert_config = bert_modeling.BertConfig(vocab_size=100, hidden_size=16, intermediate_size=32, max_position_embeddings=128, num_attention_heads=2, num_hidden_layers=1)
        bert_model = export_tfhub.create_bert_model(bert_config)
        model_checkpoint_dir = os.path.join(self.get_temp_dir(), 'checkpoint')
        checkpoint = tf.train.Checkpoint(model=bert_model)
        checkpoint.save(os.path.join(model_checkpoint_dir, 'test'))
        model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)
        vocab_file = os.path.join(self.get_temp_dir(), 'uncased_vocab.txt')
        with tf.io.gfile.GFile(vocab_file, 'w') as f:
            f.write('dummy content')
        hub_destination = os.path.join(self.get_temp_dir(), 'hub')
        export_tfhub.export_bert_tfhub(bert_config, model_checkpoint_path, hub_destination, vocab_file)
        hub_layer = hub.KerasLayer(hub_destination, trainable=True)
        if hasattr(hub_layer, 'resolved_object'):
            self.assertTrue(hub_layer.resolved_object.do_lower_case.numpy())
            with tf.io.gfile.GFile(hub_layer.resolved_object.vocab_file.asset_path.numpy()) as f:
                self.assertEqual('dummy content', f.read())
        for (source_weight, hub_weight) in zip(bert_model.trainable_weights, hub_layer.trainable_weights):
            self.assertAllClose(source_weight.numpy(), hub_weight.numpy())
        dummy_ids = np.zeros((2, 10), dtype=np.int32)
        hub_outputs = hub_layer([dummy_ids, dummy_ids, dummy_ids])
        source_outputs = bert_model([dummy_ids, dummy_ids, dummy_ids])
        self.assertEqual(hub_outputs[0].shape, (2, 16))
        self.assertEqual(hub_outputs[1].shape, (2, 10, 16))
        for (source_output, hub_output) in zip(source_outputs, hub_outputs):
            self.assertAllClose(source_output.numpy(), hub_output.numpy())
if __name__ == '__main__':
    assert tf.version.VERSION.startswith('2.')
    tf.test.main()