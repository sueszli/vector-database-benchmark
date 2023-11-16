"""Network structure for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
SUPPORTED_RNNS = {'lstm': tf.nn.rnn_cell.BasicLSTMCell, 'rnn': tf.nn.rnn_cell.RNNCell, 'gru': tf.nn.rnn_cell.GRUCell}
_BATCH_NORM_EPSILON = 1e-05
_BATCH_NORM_DECAY = 0.997
_CONV_FILTERS = 32

def batch_norm(inputs, training):
    if False:
        i = 10
        return i + 15
    'Batch normalization layer.\n\n  Note that the momentum to use will affect validation accuracy over time.\n  Batch norm has different behaviors during training/evaluation. With a large\n  momentum, the model takes longer to get a near-accurate estimation of the\n  moving mean/variance over the entire training dataset, which means we need\n  more iterations to see good evaluation results. If the training data is evenly\n  distributed over the feature space, we can also try setting a smaller momentum\n  (such as 0.1) to get good evaluation result sooner.\n\n  Args:\n    inputs: input data for batch norm layer.\n    training: a boolean to indicate if it is in training stage.\n\n  Returns:\n    tensor output from batch norm layer.\n  '
    return tf.layers.batch_normalization(inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=True, training=training)

def _conv_bn_layer(inputs, padding, filters, kernel_size, strides, layer_id, training):
    if False:
        i = 10
        return i + 15
    'Defines 2D convolutional + batch normalization layer.\n\n  Args:\n    inputs: input data for convolution layer.\n    padding: padding to be applied before convolution layer.\n    filters: an integer, number of output filters in the convolution.\n    kernel_size: a tuple specifying the height and width of the 2D convolution\n      window.\n    strides: a tuple specifying the stride length of the convolution.\n    layer_id: an integer specifying the layer index.\n    training: a boolean to indicate which stage we are in (training/eval).\n\n  Returns:\n    tensor output from the current layer.\n  '
    inputs = tf.pad(inputs, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', use_bias=False, activation=tf.nn.relu6, name='cnn_{}'.format(layer_id))
    return batch_norm(inputs, training)

def _rnn_layer(inputs, rnn_cell, rnn_hidden_size, layer_id, is_batch_norm, is_bidirectional, training):
    if False:
        return 10
    'Defines a batch normalization + rnn layer.\n\n  Args:\n    inputs: input tensors for the current layer.\n    rnn_cell: RNN cell instance to use.\n    rnn_hidden_size: an integer for the dimensionality of the rnn output space.\n    layer_id: an integer for the index of current layer.\n    is_batch_norm: a boolean specifying whether to perform batch normalization\n      on input states.\n    is_bidirectional: a boolean specifying whether the rnn layer is\n      bi-directional.\n    training: a boolean to indicate which stage we are in (training/eval).\n\n  Returns:\n    tensor output for the current layer.\n  '
    if is_batch_norm:
        inputs = batch_norm(inputs, training)
    fw_cell = rnn_cell(num_units=rnn_hidden_size, name='rnn_fw_{}'.format(layer_id))
    bw_cell = rnn_cell(num_units=rnn_hidden_size, name='rnn_bw_{}'.format(layer_id))
    if is_bidirectional:
        (outputs, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, dtype=tf.float32, swap_memory=True)
        rnn_outputs = tf.concat(outputs, -1)
    else:
        rnn_outputs = tf.nn.dynamic_rnn(fw_cell, inputs, dtype=tf.float32, swap_memory=True)
    return rnn_outputs

class DeepSpeech2(object):
    """Define DeepSpeech2 model."""

    def __init__(self, num_rnn_layers, rnn_type, is_bidirectional, rnn_hidden_size, num_classes, use_bias):
        if False:
            i = 10
            return i + 15
        "Initialize DeepSpeech2 model.\n\n    Args:\n      num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.\n      rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.\n      is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.\n      rnn_hidden_size: an integer for the number of hidden states in each unit.\n      num_classes: an integer, the number of output classes/labels.\n      use_bias: a boolean specifying whether to use bias in the last fc layer.\n    "
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias

    def __call__(self, inputs, training):
        if False:
            return 10
        inputs = _conv_bn_layer(inputs, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11), strides=(2, 2), layer_id=1, training=training)
        inputs = _conv_bn_layer(inputs, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11), strides=(2, 1), layer_id=2, training=training)
        batch_size = tf.shape(inputs)[0]
        feat_size = inputs.get_shape().as_list()[2]
        inputs = tf.reshape(inputs, [batch_size, -1, feat_size * _CONV_FILTERS])
        rnn_cell = SUPPORTED_RNNS[self.rnn_type]
        for layer_counter in xrange(self.num_rnn_layers):
            is_batch_norm = layer_counter != 0
            inputs = _rnn_layer(inputs, rnn_cell, self.rnn_hidden_size, layer_counter + 1, is_batch_norm, self.is_bidirectional, training)
        inputs = batch_norm(inputs, training)
        logits = tf.layers.dense(inputs, self.num_classes, use_bias=self.use_bias)
        return logits