"""Various implementations of sequence layers for character prediction.

A 'sequence layer' is a part of a computation graph which is responsible of
producing a sequence of characters using extracted image features. There are
many reasonable ways to implement such layers. All of them are using RNNs.
This module provides implementations which uses 'attention' mechanism to
spatially 'pool' image features and also can use a previously predicted
character to predict the next (aka auto regression).

Usage:
  Select one of available classes, e.g. Attention or use a wrapper function to
  pick one based on your requirements:
  layer_class = sequence_layers.get_layer_class(use_attention=True,
                                                use_autoregression=True)
  layer = layer_class(net, labels_one_hot, model_params, method_params)
  char_logits = layer.create_logits()
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import abc
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
    if False:
        print('Hello World!')
    'Generates orthonormal matrices with random values.\n\n  Orthonormal initialization is important for RNNs:\n    http://arxiv.org/abs/1312.6120\n    http://smerity.com/articles/2016/orthogonal_init.html\n\n  For non-square shapes the returned matrix will be semi-orthonormal: if the\n  number of columns exceeds the number of rows, then the rows are orthonormal\n  vectors; but if the number of rows exceeds the number of columns, then the\n  columns are orthonormal vectors.\n\n  We use SVD decomposition to generate an orthonormal matrix with random\n  values. The same way as it is done in the Lasagne library for Theano. Note\n  that both u and v returned by the svd are orthogonal and random. We just need\n  to pick one with the right shape.\n\n  Args:\n    shape: a shape of the tensor matrix to initialize.\n    dtype: a dtype of the initialized tensor.\n    *args: not used.\n    **kwargs: not used.\n\n  Returns:\n    An initialized tensor.\n  '
    del args
    del kwargs
    flat_shape = (shape[0], np.prod(shape[1:]))
    w = np.random.randn(*flat_shape)
    (u, _, v) = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype=dtype)
SequenceLayerParams = collections.namedtuple('SequenceLogitsParams', ['num_lstm_units', 'weight_decay', 'lstm_state_clip_value'])

class SequenceLayerBase(object):
    """A base abstruct class for all sequence layers.

  A child class has to define following methods:
    get_train_input
    get_eval_input
    unroll_cell
  """
    __metaclass__ = abc.ABCMeta

    def __init__(self, net, labels_one_hot, model_params, method_params):
        if False:
            return 10
        'Stores argument in member variable for further use.\n\n    Args:\n      net: A tensor with shape [batch_size, num_features, feature_size] which\n        contains some extracted image features.\n      labels_one_hot: An optional (can be None) ground truth labels for the\n        input features. Is a tensor with shape\n        [batch_size, seq_length, num_char_classes]\n      model_params: A namedtuple with model parameters (model.ModelParams).\n      method_params: A SequenceLayerParams instance.\n    '
        self._params = model_params
        self._mparams = method_params
        self._net = net
        self._labels_one_hot = labels_one_hot
        self._batch_size = net.get_shape().dims[0].value
        self._char_logits = {}
        regularizer = slim.l2_regularizer(self._mparams.weight_decay)
        self._softmax_w = slim.model_variable('softmax_w', [self._mparams.num_lstm_units, self._params.num_char_classes], initializer=orthogonal_initializer, regularizer=regularizer)
        self._softmax_b = slim.model_variable('softmax_b', [self._params.num_char_classes], initializer=tf.zeros_initializer(), regularizer=regularizer)

    @abc.abstractmethod
    def get_train_input(self, prev, i):
        if False:
            for i in range(10):
                print('nop')
        'Returns a sample to be used to predict a character during training.\n\n    This function is used as a loop_function for an RNN decoder.\n\n    Args:\n      prev: output tensor from previous step of the RNN. A tensor with shape:\n        [batch_size, num_char_classes].\n      i: index of a character in the output sequence.\n\n    Returns:\n      A tensor with shape [batch_size, ?] - depth depends on implementation\n      details.\n    '
        pass

    @abc.abstractmethod
    def get_eval_input(self, prev, i):
        if False:
            i = 10
            return i + 15
        'Returns a sample to be used to predict a character during inference.\n\n    This function is used as a loop_function for an RNN decoder.\n\n    Args:\n      prev: output tensor from previous step of the RNN. A tensor with shape:\n        [batch_size, num_char_classes].\n      i: index of a character in the output sequence.\n\n    Returns:\n      A tensor with shape [batch_size, ?] - depth depends on implementation\n      details.\n    '
        raise AssertionError('Not implemented')

    @abc.abstractmethod
    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        if False:
            while True:
                i = 10
        'Unrolls an RNN cell for all inputs.\n\n    This is a placeholder to call some RNN decoder. It has a similar to\n    tf.seq2seq.rnn_decode interface.\n\n    Args:\n      decoder_inputs: A list of 2D Tensors* [batch_size x input_size]. In fact,\n        most of existing decoders in presence of a loop_function use only the\n        first element to determine batch_size and length of the list to\n        determine number of steps.\n      initial_state: 2D Tensor with shape [batch_size x cell.state_size].\n      loop_function: function will be applied to the i-th output in order to\n        generate the i+1-st input (see self.get_input).\n      cell: rnn_cell.RNNCell defining the cell function and size.\n\n    Returns:\n      A tuple of the form (outputs, state), where:\n        outputs: A list of character logits of the same length as\n        decoder_inputs of 2D Tensors with shape [batch_size x num_characters].\n        state: The state of each cell at the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n    '
        pass

    def is_training(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the layer is created for training stage.'
        return self._labels_one_hot is not None

    def char_logit(self, inputs, char_index):
        if False:
            for i in range(10):
                print('nop')
        'Creates logits for a character if required.\n\n    Args:\n      inputs: A tensor with shape [batch_size, ?] (depth is implementation\n        dependent).\n      char_index: A integer index of a character in the output sequence.\n\n    Returns:\n      A tensor with shape [batch_size, num_char_classes]\n    '
        if char_index not in self._char_logits:
            self._char_logits[char_index] = tf.nn.xw_plus_b(inputs, self._softmax_w, self._softmax_b)
        return self._char_logits[char_index]

    def char_one_hot(self, logit):
        if False:
            for i in range(10):
                print('nop')
        'Creates one hot encoding for a logit of a character.\n\n    Args:\n      logit: A tensor with shape [batch_size, num_char_classes].\n\n    Returns:\n      A tensor with shape [batch_size, num_char_classes]\n    '
        prediction = tf.argmax(logit, axis=1)
        return slim.one_hot_encoding(prediction, self._params.num_char_classes)

    def get_input(self, prev, i):
        if False:
            while True:
                i = 10
        'A wrapper for get_train_input and get_eval_input.\n\n    Args:\n      prev: output tensor from previous step of the RNN. A tensor with shape:\n        [batch_size, num_char_classes].\n      i: index of a character in the output sequence.\n\n    Returns:\n      A tensor with shape [batch_size, ?] - depth depends on implementation\n      details.\n    '
        if self.is_training():
            return self.get_train_input(prev, i)
        else:
            return self.get_eval_input(prev, i)

    def create_logits(self):
        if False:
            print('Hello World!')
        'Creates character sequence logits for a net specified in the constructor.\n\n    A "main" method for the sequence layer which glues together all pieces.\n\n    Returns:\n      A tensor with shape [batch_size, seq_length, num_char_classes].\n    '
        with tf.variable_scope('LSTM'):
            first_label = self.get_input(prev=None, i=0)
            decoder_inputs = [first_label] + [None] * (self._params.seq_length - 1)
            lstm_cell = tf.contrib.rnn.LSTMCell(self._mparams.num_lstm_units, use_peepholes=False, cell_clip=self._mparams.lstm_state_clip_value, state_is_tuple=True, initializer=orthogonal_initializer)
            (lstm_outputs, _) = self.unroll_cell(decoder_inputs=decoder_inputs, initial_state=lstm_cell.zero_state(self._batch_size, tf.float32), loop_function=self.get_input, cell=lstm_cell)
        with tf.variable_scope('logits'):
            logits_list = [tf.expand_dims(self.char_logit(logit, i), dim=1) for (i, logit) in enumerate(lstm_outputs)]
        return tf.concat(logits_list, 1)

class NetSlice(SequenceLayerBase):
    """A layer which uses a subset of image features to predict each character.
  """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(NetSlice, self).__init__(*args, **kwargs)
        self._zero_label = tf.zeros([self._batch_size, self._params.num_char_classes])

    def get_image_feature(self, char_index):
        if False:
            for i in range(10):
                print('nop')
        'Returns a subset of image features for a character.\n\n    Args:\n      char_index: an index of a character.\n\n    Returns:\n      A tensor with shape [batch_size, ?]. The output depth depends on the\n      depth of input net.\n    '
        (batch_size, features_num, _) = [d.value for d in self._net.get_shape()]
        slice_len = int(features_num / self._params.seq_length)
        net_slice = self._net[:, char_index:char_index + slice_len, :]
        feature = tf.reshape(net_slice, [batch_size, -1])
        logging.debug('Image feature: %s', feature)
        return feature

    def get_eval_input(self, prev, i):
        if False:
            return 10
        'See SequenceLayerBase.get_eval_input for details.'
        del prev
        return self.get_image_feature(i)

    def get_train_input(self, prev, i):
        if False:
            while True:
                i = 10
        'See SequenceLayerBase.get_train_input for details.'
        return self.get_eval_input(prev, i)

    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        if False:
            while True:
                i = 10
        'See SequenceLayerBase.unroll_cell for details.'
        return tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs, initial_state=initial_state, cell=cell, loop_function=self.get_input)

class NetSliceWithAutoregression(NetSlice):
    """A layer similar to NetSlice, but it also uses auto regression.

  The "auto regression" means that we use network output for previous character
  as a part of input for the current character.
  """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(NetSliceWithAutoregression, self).__init__(*args, **kwargs)

    def get_eval_input(self, prev, i):
        if False:
            print('Hello World!')
        'See SequenceLayerBase.get_eval_input for details.'
        if i == 0:
            prev = self._zero_label
        else:
            logit = self.char_logit(prev, char_index=i - 1)
            prev = self.char_one_hot(logit)
        image_feature = self.get_image_feature(char_index=i)
        return tf.concat([image_feature, prev], 1)

    def get_train_input(self, prev, i):
        if False:
            while True:
                i = 10
        'See SequenceLayerBase.get_train_input for details.'
        if i == 0:
            prev = self._zero_label
        else:
            prev = self._labels_one_hot[:, i - 1, :]
        image_feature = self.get_image_feature(i)
        return tf.concat([image_feature, prev], 1)

class Attention(SequenceLayerBase):
    """A layer which uses attention mechanism to select image features."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Attention, self).__init__(*args, **kwargs)
        self._zero_label = tf.zeros([self._batch_size, self._params.num_char_classes])

    def get_eval_input(self, prev, i):
        if False:
            i = 10
            return i + 15
        'See SequenceLayerBase.get_eval_input for details.'
        del prev, i
        return self._zero_label

    def get_train_input(self, prev, i):
        if False:
            return 10
        'See SequenceLayerBase.get_train_input for details.'
        return self.get_eval_input(prev, i)

    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        if False:
            return 10
        return tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs, initial_state=initial_state, attention_states=self._net, cell=cell, loop_function=self.get_input)

class AttentionWithAutoregression(Attention):
    """A layer which uses both attention and auto regression."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(AttentionWithAutoregression, self).__init__(*args, **kwargs)

    def get_train_input(self, prev, i):
        if False:
            print('Hello World!')
        'See SequenceLayerBase.get_train_input for details.'
        if i == 0:
            return self._zero_label
        else:
            return self._labels_one_hot[:, i - 1, :]

    def get_eval_input(self, prev, i):
        if False:
            print('Hello World!')
        'See SequenceLayerBase.get_eval_input for details.'
        if i == 0:
            return self._zero_label
        else:
            logit = self.char_logit(prev, char_index=i - 1)
            return self.char_one_hot(logit)

def get_layer_class(use_attention, use_autoregression):
    if False:
        return 10
    'A convenience function to get a layer class based on requirements.\n\n  Args:\n    use_attention: if True a returned class will use attention.\n    use_autoregression: if True a returned class will use auto regression.\n\n  Returns:\n    One of available sequence layers (child classes for SequenceLayerBase).\n  '
    if use_attention and use_autoregression:
        layer_class = AttentionWithAutoregression
    elif use_attention and (not use_autoregression):
        layer_class = Attention
    elif not use_attention and (not use_autoregression):
        layer_class = NetSlice
    elif not use_attention and use_autoregression:
        layer_class = NetSliceWithAutoregression
    else:
        raise AssertionError('Unsupported sequence layer class')
    logging.debug('Use %s as a layer class', layer_class.__name__)
    return layer_class