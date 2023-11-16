"""Zoneout Wrapper"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class ZoneoutWrapper(tf.contrib.rnn.RNNCell):
    """Add Zoneout to a RNN cell."""

    def __init__(self, cell, zoneout_drop_prob, is_training=True):
        if False:
            print('Hello World!')
        self._cell = cell
        self._zoneout_prob = zoneout_drop_prob
        self._is_training = is_training

    @property
    def state_size(self):
        if False:
            i = 10
            return i + 15
        return self._cell.state_size

    @property
    def output_size(self):
        if False:
            while True:
                i = 10
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if False:
            for i in range(10):
                print('nop')
        (output, new_state) = self._cell(inputs, state, scope)
        if not isinstance(self._cell.state_size, tuple):
            new_state = tf.split(value=new_state, num_or_size_splits=2, axis=1)
            state = tf.split(value=state, num_or_size_splits=2, axis=1)
        final_new_state = [new_state[0], new_state[1]]
        if self._is_training:
            for (i, state_element) in enumerate(state):
                random_tensor = 1 - self._zoneout_prob
                random_tensor += tf.random_uniform(tf.shape(state_element))
                binary_tensor = tf.floor(random_tensor)
                final_new_state[i] = (new_state[i] - state_element) * binary_tensor + state_element
        else:
            for (i, state_element) in enumerate(state):
                final_new_state[i] = state_element * self._zoneout_prob + new_state[i] * (1 - self._zoneout_prob)
        if isinstance(self._cell.state_size, tuple):
            return (output, tf.contrib.rnn.LSTMStateTuple(final_new_state[0], final_new_state[1]))
        return (output, tf.concat([final_new_state[0], final_new_state[1]], 1))