"""Model for sentiment analysis.

The model makes use of concatenation of two CNN layers with
different kernel sizes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class CNN(tf.keras.models.Model):
    """CNN for sentimental analysis."""

    def __init__(self, emb_dim, num_words, sentence_length, hid_dim, class_dim, dropout_rate):
        if False:
            for i in range(10):
                print('nop')
        'Initialize CNN model.\n\n    Args:\n      emb_dim: The dimension of the Embedding layer.\n      num_words: The number of the most frequent tokens\n        to be used from the corpus.\n      sentence_length: The number of words in each sentence.\n        Longer sentences get cut, shorter ones padded.\n      hid_dim: The dimension of the Embedding layer.\n      class_dim: The number of the CNN layer filters.\n      dropout_rate: The portion of kept value in the Dropout layer.\n    Returns:\n      tf.keras.models.Model: A Keras model.\n    '
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer = tf.keras.layers.Embedding(num_words, output_dim=emb_dim)(input_layer)
        layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation='relu')(layer)
        layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)
        layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation='relu')(layer)
        layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)
        layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        output = tf.keras.layers.Dense(class_dim, activation='softmax')(layer)
        super(CNN, self).__init__(inputs=[input_layer], outputs=output)