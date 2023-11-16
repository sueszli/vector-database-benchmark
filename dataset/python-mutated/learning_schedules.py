"""Library of common learning rate schedules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

def exponential_decay_with_burnin(global_step, learning_rate_base, learning_rate_decay_steps, learning_rate_decay_factor, burnin_learning_rate=0.0, burnin_steps=0, min_learning_rate=0.0, staircase=True):
    if False:
        print('Hello World!')
    'Exponential decay schedule with burn-in period.\n\n  In this schedule, learning rate is fixed at burnin_learning_rate\n  for a fixed period, before transitioning to a regular exponential\n  decay schedule.\n\n  Args:\n    global_step: int tensor representing global step.\n    learning_rate_base: base learning rate.\n    learning_rate_decay_steps: steps to take between decaying the learning rate.\n      Note that this includes the number of burn-in steps.\n    learning_rate_decay_factor: multiplicative factor by which to decay\n      learning rate.\n    burnin_learning_rate: initial learning rate during burn-in period.  If\n      0.0 (which is the default), then the burn-in learning rate is simply\n      set to learning_rate_base.\n    burnin_steps: number of steps to use burnin learning rate.\n    min_learning_rate: the minimum learning rate.\n    staircase: whether use staircase decay.\n\n  Returns:\n    If executing eagerly:\n      returns a no-arg callable that outputs the (scalar)\n      float tensor learning rate given the current value of global_step.\n    If in a graph:\n      immediately returns a (scalar) float tensor representing learning rate.\n  '
    if burnin_learning_rate == 0:
        burnin_learning_rate = learning_rate_base

    def eager_decay_rate():
        if False:
            return 10
        'Callable to compute the learning rate.'
        post_burnin_learning_rate = tf.train.exponential_decay(learning_rate_base, global_step - burnin_steps, learning_rate_decay_steps, learning_rate_decay_factor, staircase=staircase)
        if callable(post_burnin_learning_rate):
            post_burnin_learning_rate = post_burnin_learning_rate()
        return tf.maximum(tf.where(tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)), tf.constant(burnin_learning_rate), post_burnin_learning_rate), min_learning_rate, name='learning_rate')
    if tf.executing_eagerly():
        return eager_decay_rate
    else:
        return eager_decay_rate()

def cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0, warmup_steps=0, hold_base_rate_steps=0):
    if False:
        return 10
    'Cosine decay schedule with warm up period.\n\n  Cosine annealing learning rate as described in:\n    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.\n    ICLR 2017. https://arxiv.org/abs/1608.03983\n  In this schedule, the learning rate grows linearly from warmup_learning_rate\n  to learning_rate_base for warmup_steps, then transitions to a cosine decay\n  schedule.\n\n  Args:\n    global_step: int64 (scalar) tensor representing global step.\n    learning_rate_base: base learning rate.\n    total_steps: total number of training steps.\n    warmup_learning_rate: initial learning rate for warm up.\n    warmup_steps: number of warmup steps.\n    hold_base_rate_steps: Optional number of steps to hold base learning rate\n      before decaying.\n\n  Returns:\n    If executing eagerly:\n      returns a no-arg callable that outputs the (scalar)\n      float tensor learning rate given the current value of global_step.\n    If in a graph:\n      immediately returns a (scalar) float tensor representing learning rate.\n\n  Raises:\n    ValueError: if warmup_learning_rate is larger than learning_rate_base,\n      or if warmup_steps is larger than total_steps.\n  '
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')

    def eager_decay_rate():
        if False:
            while True:
                i = 10
        'Callable to compute the learning rate.'
        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(np.pi * (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * tf.cast(global_step, tf.float32) + warmup_learning_rate
            learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)
        return tf.where(global_step > total_steps, 0.0, learning_rate, name='learning_rate')
    if tf.executing_eagerly():
        return eager_decay_rate
    else:
        return eager_decay_rate()

def manual_stepping(global_step, boundaries, rates, warmup=False):
    if False:
        print('Hello World!')
    'Manually stepped learning rate schedule.\n\n  This function provides fine grained control over learning rates.  One must\n  specify a sequence of learning rates as well as a set of integer steps\n  at which the current learning rate must transition to the next.  For example,\n  if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning\n  rate returned by this function is .1 for global_step=0,...,4, .01 for\n  global_step=5...9, and .001 for global_step=10 and onward.\n\n  Args:\n    global_step: int64 (scalar) tensor representing global step.\n    boundaries: a list of global steps at which to switch learning\n      rates.  This list is assumed to consist of increasing positive integers.\n    rates: a list of (float) learning rates corresponding to intervals between\n      the boundaries.  The length of this list must be exactly\n      len(boundaries) + 1.\n    warmup: Whether to linearly interpolate learning rate for steps in\n      [0, boundaries[0]].\n\n  Returns:\n    If executing eagerly:\n      returns a no-arg callable that outputs the (scalar)\n      float tensor learning rate given the current value of global_step.\n    If in a graph:\n      immediately returns a (scalar) float tensor representing learning rate.\n  Raises:\n    ValueError: if one of the following checks fails:\n      1. boundaries is a strictly increasing list of positive integers\n      2. len(rates) == len(boundaries) + 1\n      3. boundaries[0] != 0\n  '
    if any([b < 0 for b in boundaries]) or any([not isinstance(b, int) for b in boundaries]):
        raise ValueError('boundaries must be a list of positive integers')
    if any([bnext <= b for (bnext, b) in zip(boundaries[1:], boundaries[:-1])]):
        raise ValueError('Entries in boundaries must be strictly increasing.')
    if any([not isinstance(r, float) for r in rates]):
        raise ValueError('Learning rates must be floats')
    if len(rates) != len(boundaries) + 1:
        raise ValueError('Number of provided learning rates must exceed number of boundary points by exactly 1.')
    if boundaries and boundaries[0] == 0:
        raise ValueError('First step cannot be zero.')
    if warmup and boundaries:
        slope = (rates[1] - rates[0]) * 1.0 / boundaries[0]
        warmup_steps = list(range(boundaries[0]))
        warmup_rates = [rates[0] + slope * step for step in warmup_steps]
        boundaries = warmup_steps + boundaries
        rates = warmup_rates + rates[1:]
    else:
        boundaries = [0] + boundaries
    num_boundaries = len(boundaries)

    def eager_decay_rate():
        if False:
            print('Hello World!')
        'Callable to compute the learning rate.'
        rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries), list(range(num_boundaries)), [0] * num_boundaries))
        return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=num_boundaries), name='learning_rate')
    if tf.executing_eagerly():
        return eager_decay_rate
    else:
        return eager_decay_rate()