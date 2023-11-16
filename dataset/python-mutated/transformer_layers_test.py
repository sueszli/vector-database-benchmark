"""Tests for layers in Transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.transformer.v2 import attention_layer
from official.transformer.v2 import embedding_layer
from official.transformer.v2 import ffn_layer
from official.transformer.v2 import metrics

class TransformerLayersTest(tf.test.TestCase):

    def test_attention_layer(self):
        if False:
            return 10
        hidden_size = 64
        num_heads = 4
        dropout = 0.5
        dim_per_head = hidden_size // num_heads
        layer = attention_layer.SelfAttention(hidden_size, num_heads, dropout)
        self.assertDictEqual(layer.get_config(), {'hidden_size': hidden_size, 'num_heads': num_heads, 'attention_dropout': dropout})
        length = 2
        x = tf.ones([1, length, hidden_size])
        bias = tf.ones([1])
        cache = {'k': tf.zeros([1, 0, num_heads, dim_per_head]), 'v': tf.zeros([1, 0, num_heads, dim_per_head])}
        y = layer(x, bias, training=True, cache=cache)
        self.assertEqual(y.shape, (1, length, 64))
        self.assertEqual(cache['k'].shape, (1, length, num_heads, dim_per_head))
        self.assertEqual(cache['v'].shape, (1, length, num_heads, dim_per_head))

    def test_embedding_shared_weights(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_size = 50
        hidden_size = 64
        length = 2
        layer = embedding_layer.EmbeddingSharedWeights(vocab_size, hidden_size)
        self.assertDictEqual(layer.get_config(), {'vocab_size': 50, 'hidden_size': 64})
        idx = tf.ones([1, length], dtype='int32')
        y = layer(idx)
        self.assertEqual(y.shape, (1, length, hidden_size))
        x = tf.ones([1, length, hidden_size])
        output = layer(x, 'linear')
        self.assertEqual(output.shape, (1, length, vocab_size))

    def test_feed_forward_network(self):
        if False:
            while True:
                i = 10
        hidden_size = 64
        filter_size = 32
        relu_dropout = 0.5
        layer = ffn_layer.FeedForwardNetwork(hidden_size, filter_size, relu_dropout)
        self.assertDictEqual(layer.get_config(), {'hidden_size': hidden_size, 'filter_size': filter_size, 'relu_dropout': relu_dropout})
        length = 2
        x = tf.ones([1, length, hidden_size])
        y = layer(x, training=True)
        self.assertEqual(y.shape, (1, length, hidden_size))

    def test_metric_layer(self):
        if False:
            while True:
                i = 10
        vocab_size = 50
        logits = tf.keras.layers.Input((None, vocab_size), dtype='float32', name='logits')
        targets = tf.keras.layers.Input((None,), dtype='int64', name='targets')
        output_logits = metrics.MetricLayer(vocab_size)([logits, targets])
        self.assertEqual(output_logits.shape.as_list(), [None, None, vocab_size])
if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()