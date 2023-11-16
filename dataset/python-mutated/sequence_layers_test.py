"""Tests for sequence_layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import model
import sequence_layers

def fake_net(batch_size, num_features, feature_size):
    if False:
        return 10
    return tf.convert_to_tensor(np.random.uniform(size=(batch_size, num_features, feature_size)), dtype=tf.float32)

def fake_labels(batch_size, seq_length, num_char_classes):
    if False:
        print('Hello World!')
    labels_np = tf.convert_to_tensor(np.random.randint(low=0, high=num_char_classes, size=(batch_size, seq_length)))
    return slim.one_hot_encoding(labels_np, num_classes=num_char_classes)

def create_layer(layer_class, batch_size, seq_length, num_char_classes):
    if False:
        while True:
            i = 10
    model_params = model.ModelParams(num_char_classes=num_char_classes, seq_length=seq_length, num_views=1, null_code=num_char_classes)
    net = fake_net(batch_size=batch_size, num_features=seq_length * 5, feature_size=6)
    labels_one_hot = fake_labels(batch_size, seq_length, num_char_classes)
    layer_params = sequence_layers.SequenceLayerParams(num_lstm_units=10, weight_decay=4e-05, lstm_state_clip_value=10.0)
    return layer_class(net, labels_one_hot, model_params, layer_params)

class SequenceLayersTest(tf.test.TestCase):

    def test_net_slice_char_logits_with_correct_shape(self):
        if False:
            return 10
        batch_size = 2
        seq_length = 4
        num_char_classes = 3
        layer = create_layer(sequence_layers.NetSlice, batch_size, seq_length, num_char_classes)
        char_logits = layer.create_logits()
        self.assertEqual(tf.TensorShape([batch_size, seq_length, num_char_classes]), char_logits.get_shape())

    def test_net_slice_with_autoregression_char_logits_with_correct_shape(self):
        if False:
            while True:
                i = 10
        batch_size = 2
        seq_length = 4
        num_char_classes = 3
        layer = create_layer(sequence_layers.NetSliceWithAutoregression, batch_size, seq_length, num_char_classes)
        char_logits = layer.create_logits()
        self.assertEqual(tf.TensorShape([batch_size, seq_length, num_char_classes]), char_logits.get_shape())

    def test_attention_char_logits_with_correct_shape(self):
        if False:
            return 10
        batch_size = 2
        seq_length = 4
        num_char_classes = 3
        layer = create_layer(sequence_layers.Attention, batch_size, seq_length, num_char_classes)
        char_logits = layer.create_logits()
        self.assertEqual(tf.TensorShape([batch_size, seq_length, num_char_classes]), char_logits.get_shape())

    def test_attention_with_autoregression_char_logits_with_correct_shape(self):
        if False:
            while True:
                i = 10
        batch_size = 2
        seq_length = 4
        num_char_classes = 3
        layer = create_layer(sequence_layers.AttentionWithAutoregression, batch_size, seq_length, num_char_classes)
        char_logits = layer.create_logits()
        self.assertEqual(tf.TensorShape([batch_size, seq_length, num_char_classes]), char_logits.get_shape())
if __name__ == '__main__':
    tf.test.main()