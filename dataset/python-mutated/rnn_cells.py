"""Custom RNN cells for hierarchical RNNs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from learned_optimizer.optimizer import utils

class BiasGRUCell(tf.contrib.rnn.RNNCell):
    """GRU cell (cf. http://arxiv.org/abs/1406.1078) with an additional bias."""

    def __init__(self, num_units, activation=tf.tanh, scale=0.1, gate_bias_init=0.0, random_seed=None):
        if False:
            for i in range(10):
                print('nop')
        self._num_units = num_units
        self._activation = activation
        self._scale = scale
        self._gate_bias_init = gate_bias_init
        self._random_seed = random_seed

    @property
    def state_size(self):
        if False:
            return 10
        return self._num_units

    @property
    def output_size(self):
        if False:
            print('Hello World!')
        return self._num_units

    def __call__(self, inputs, state, bias=None):
        if False:
            i = 10
            return i + 15
        if bias is None:
            bias = tf.zeros((1, 3))
        (r_bias, u_bias, c_bias) = tf.split(bias, 3, 1)
        with tf.variable_scope(type(self).__name__):
            with tf.variable_scope('gates'):
                proj = utils.affine([inputs, state], 2 * self._num_units, scale=self._scale, bias_init=self._gate_bias_init, random_seed=self._random_seed)
                (r_lin, u_lin) = tf.split(proj, 2, 1)
                (r, u) = (tf.nn.sigmoid(r_lin + r_bias), tf.nn.sigmoid(u_lin + u_bias))
            with tf.variable_scope('candidate'):
                proj = utils.affine([inputs, r * state], self._num_units, scale=self._scale, random_seed=self._random_seed)
                c = self._activation(proj + c_bias)
            new_h = u * state + (1 - u) * c
        return (new_h, new_h)