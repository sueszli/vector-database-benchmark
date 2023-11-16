"""Context functions.

Given the current contexts, timer and context sampler, returns new contexts
  after an environment step. This can be used to define a high-level policy
  that controls contexts as its actions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin.tf
import utils as uvf_utils

@gin.configurable
def periodic_context_fn(contexts, timer, sampler_fn, period=1):
    if False:
        print('Hello World!')
    'Periodically samples contexts.\n\n  Args:\n    contexts: a list of [num_context_dims] tensor variables representing\n      current contexts.\n    timer: a scalar integer tensor variable holding the current time step.\n    sampler_fn: a sampler function that samples a list of [num_context_dims]\n      tensors.\n    period: (integer) period of update.\n  Returns:\n    a list of [num_context_dims] tensors.\n  '
    contexts = list(contexts[:])
    return tf.cond(tf.mod(timer, period) == 0, sampler_fn, lambda : contexts)

@gin.configurable
def timer_context_fn(contexts, timer, sampler_fn, period=1, timer_index=-1, debug=False):
    if False:
        for i in range(10):
            print('nop')
    'Samples contexts based on timer in contexts.\n\n  Args:\n    contexts: a list of [num_context_dims] tensor variables representing\n      current contexts.\n    timer: a scalar integer tensor variable holding the current time step.\n    sampler_fn: a sampler function that samples a list of [num_context_dims]\n      tensors.\n    period: (integer) period of update; actual period = `period` + 1.\n    timer_index: (integer) Index of context list that present timer.\n    debug: (boolean) Print debug messages.\n  Returns:\n    a list of [num_context_dims] tensors.\n  '
    contexts = list(contexts[:])
    cond = tf.equal(contexts[timer_index][0], 0)

    def reset():
        if False:
            for i in range(10):
                print('nop')
        'Sample context and reset the timer.'
        new_contexts = sampler_fn()
        new_contexts[timer_index] = tf.zeros_like(contexts[timer_index]) + period
        return new_contexts

    def update():
        if False:
            print('Hello World!')
        'Decrement the timer.'
        contexts[timer_index] -= 1
        return contexts
    values = tf.cond(cond, reset, update)
    if debug:
        values[0] = uvf_utils.tf_print(values[0], values + [timer], 'timer_context_fn', first_n=200, name='timer_context_fn:contexts')
    return values

@gin.configurable
def relative_context_transition_fn(contexts, timer, sampler_fn, k=2, state=None, next_state=None, **kwargs):
    if False:
        print('Hello World!')
    'Contexts updated to be relative to next state.\n  '
    contexts = list(contexts[:])
    assert len(contexts) == 1
    new_contexts = [tf.concat([contexts[0][:k] + state[:k] - next_state[:k], contexts[0][k:]], -1)]
    return new_contexts

@gin.configurable
def relative_context_multi_transition_fn(contexts, timer, sampler_fn, k=2, states=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Given contexts at first state and sequence of states, derives sequence of all contexts.\n  '
    contexts = list(contexts[:])
    assert len(contexts) == 1
    contexts = [tf.concat([tf.expand_dims(contexts[0][:, :k] + states[:, 0, :k], 1) - states[:, :, :k], contexts[0][:, None, k:] * tf.ones_like(states[:, :, :1])], -1)]
    return contexts