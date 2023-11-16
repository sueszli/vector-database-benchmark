"""Implementation of embedding layer with shared weights."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        if False:
            print('Hello World!')
        'Specify characteristic parameters of embedding layer.\n\n    Args:\n      vocab_size: Number of tokens in the embedding. (Typically ~32,000)\n      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)\n    '
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        'Build embedding layer.'
        with tf.name_scope('embedding_and_softmax'):
            self.shared_weights = self.add_weight('weights', shape=[self.vocab_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=self.hidden_size ** (-0.5)))
        super(EmbeddingSharedWeights, self).build(input_shape)

    def get_config(self):
        if False:
            return 10
        return {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size}

    def call(self, inputs, mode='embedding'):
        if False:
            i = 10
            return i + 15
        'Get token embeddings of inputs.\n\n    Args:\n      inputs: An int64 tensor with shape [batch_size, length]\n      mode: string, a valid value is one of "embedding" and "linear".\n    Returns:\n      outputs: (1) If mode == "embedding", output embedding tensor, float32 with\n        shape [batch_size, length, embedding_size]; (2) mode == "linear", output\n        linear tensor, float32 with shape [batch_size, length, vocab_size].\n    Raises:\n      ValueError: if mode is not valid.\n    '
        if mode == 'embedding':
            return self._embedding(inputs)
        elif mode == 'linear':
            return self._linear(inputs)
        else:
            raise ValueError('mode {} is not valid.'.format(mode))

    def _embedding(self, inputs):
        if False:
            i = 10
            return i + 15
        'Applies embedding based on inputs tensor.'
        with tf.name_scope('embedding'):
            embeddings = tf.gather(self.shared_weights, inputs)
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)
            embeddings *= self.hidden_size ** 0.5
            return embeddings

    def _linear(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Computes logits by running inputs through a linear layer.\n\n    Args:\n      inputs: A float32 tensor with shape [batch_size, length, hidden_size]\n    Returns:\n      float32 tensor with shape [batch_size, length, vocab_size].\n    '
        with tf.name_scope('presoftmax_linear'):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]
            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)
            return tf.reshape(logits, [batch_size, length, self.vocab_size])