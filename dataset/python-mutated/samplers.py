"""Samplers for Contexts.

  Each sampler class should define __call__(batch_size).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import gin.tf

@gin.configurable
class BaseSampler(object):
    """Base sampler."""

    def __init__(self, context_spec, context_range=None, k=2, scope='sampler'):
        if False:
            while True:
                i = 10
        'Construct a base sampler.\n\n    Args:\n      context_spec: A context spec.\n      context_range: A tuple of (minval, max), where minval, maxval are floats\n        or Numpy arrays with the same shape as the context.\n      scope: A string denoting scope.\n    '
        self._context_spec = context_spec
        self._context_range = context_range
        self._k = k
        self._scope = scope

    def __call__(self, batch_size, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def set_replay(self, replay=None):
        if False:
            i = 10
            return i + 15
        pass

    def _validate_contexts(self, contexts):
        if False:
            while True:
                i = 10
        'Validate if contexts have right spec.\n\n    Args:\n      contexts: A [batch_size, num_contexts_dim] tensor.\n    Raises:\n      ValueError: If shape or dtype mismatches that of spec.\n    '
        if contexts[0].shape != self._context_spec.shape:
            raise ValueError('contexts has invalid shape %s wrt spec shape %s' % (contexts[0].shape, self._context_spec.shape))
        if contexts.dtype != self._context_spec.dtype:
            raise ValueError('contexts has invalid dtype %s wrt spec dtype %s' % (contexts.dtype, self._context_spec.dtype))

@gin.configurable
class ZeroSampler(BaseSampler):
    """Zero sampler."""

    def __call__(self, batch_size, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        contexts = tf.zeros(dtype=self._context_spec.dtype, shape=[batch_size] + self._context_spec.shape.as_list())
        return (contexts, contexts)

@gin.configurable
class BinarySampler(BaseSampler):
    """Binary sampler."""

    def __init__(self, probs=0.5, *args, **kwargs):
        if False:
            print('Hello World!')
        'Constructor.'
        super(BinarySampler, self).__init__(*args, **kwargs)
        self._probs = probs

    def __call__(self, batch_size, **kwargs):
        if False:
            i = 10
            return i + 15
        'Sample a batch of context.'
        spec = self._context_spec
        contexts = tf.random_uniform(shape=[batch_size] + spec.shape.as_list(), dtype=tf.float32)
        contexts = tf.cast(tf.greater(contexts, self._probs), dtype=spec.dtype)
        return (contexts, contexts)

@gin.configurable
class RandomSampler(BaseSampler):
    """Random sampler."""

    def __call__(self, batch_size, **kwargs):
        if False:
            return 10
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        spec = self._context_spec
        context_range = self._context_range
        if isinstance(context_range[0], (int, float)):
            contexts = tf.random_uniform(shape=[batch_size] + spec.shape.as_list(), minval=context_range[0], maxval=context_range[1], dtype=spec.dtype)
        elif isinstance(context_range[0], (list, tuple, np.ndarray)):
            assert len(spec.shape.as_list()) == 1
            assert spec.shape.as_list()[0] == len(context_range[0])
            assert spec.shape.as_list()[0] == len(context_range[1])
            contexts = tf.concat([tf.random_uniform(shape=[batch_size, 1] + spec.shape.as_list()[1:], minval=context_range[0][i], maxval=context_range[1][i], dtype=spec.dtype) for i in range(spec.shape.as_list()[0])], axis=1)
        else:
            raise NotImplementedError(context_range)
        self._validate_contexts(contexts)
        (state, next_state) = (kwargs['state'], kwargs['next_state'])
        if state is not None and next_state is not None:
            pass
        return (contexts, contexts)

@gin.configurable
class ScheduledSampler(BaseSampler):
    """Scheduled sampler."""

    def __init__(self, scope='default', values=None, scheduler='cycle', scheduler_params=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Construct sampler.\n\n    Args:\n      scope: Scope name.\n      values: A list of numbers or [num_context_dim] Numpy arrays\n        representing the values to cycle.\n      scheduler: scheduler type.\n      scheduler_params: scheduler parameters.\n      *args: arguments.\n      **kwargs: keyword arguments.\n    '
        super(ScheduledSampler, self).__init__(*args, **kwargs)
        self._scope = scope
        self._values = values
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params or {}
        assert self._values is not None and len(self._values), 'must provide non-empty values.'
        self._n = len(self._values)
        self._count = 0
        self._i = tf.Variable(tf.zeros(shape=(), dtype=tf.int32), name='%s-scheduled_sampler_%d' % (self._scope, self._count))
        self._values = tf.constant(self._values, dtype=self._context_spec.dtype)

    def __call__(self, batch_size, **kwargs):
        if False:
            while True:
                i = 10
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        spec = self._context_spec
        next_op = self._next(self._i)
        with tf.control_dependencies([next_op]):
            value = self._values[self._i]
            if value.get_shape().as_list():
                values = tf.tile(tf.expand_dims(value, 0), (batch_size,) + (1,) * spec.shape.ndims)
            else:
                values = value + tf.zeros(shape=[batch_size] + spec.shape.as_list(), dtype=spec.dtype)
        self._validate_contexts(values)
        self._count += 1
        return (values, values)

    def _next(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Return op that increments pointer to next value.\n\n    Args:\n      i: A tensorflow integer variable.\n    Returns:\n      Op that increments pointer.\n    '
        if self._scheduler == 'cycle':
            inc = 'inc' in self._scheduler_params and self._scheduler_params['inc'] or 1
            return tf.assign(i, tf.mod(i + inc, self._n))
        else:
            raise NotImplementedError(self._scheduler)

@gin.configurable
class ReplaySampler(BaseSampler):
    """Replay sampler."""

    def __init__(self, prefetch_queue_capacity=2, override_indices=None, state_indices=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Construct sampler.\n\n    Args:\n      prefetch_queue_capacity: Capacity for prefetch queue.\n      override_indices: Override indices.\n      state_indices: Select certain indices from state dimension.\n      *args: arguments.\n      **kwargs: keyword arguments.\n    '
        super(ReplaySampler, self).__init__(*args, **kwargs)
        self._prefetch_queue_capacity = prefetch_queue_capacity
        self._override_indices = override_indices
        self._state_indices = state_indices

    def set_replay(self, replay):
        if False:
            for i in range(10):
                print('nop')
        'Set replay.\n\n    Args:\n      replay: A replay buffer.\n    '
        self._replay = replay

    def __call__(self, batch_size, **kwargs):
        if False:
            print('Hello World!')
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        batch = self._replay.GetRandomBatch(batch_size)
        next_states = batch[4]
        if self._prefetch_queue_capacity > 0:
            batch_queue = slim.prefetch_queue.prefetch_queue([next_states], capacity=self._prefetch_queue_capacity, name='%s/batch_context_queue' % self._scope)
            next_states = batch_queue.dequeue()
        if self._override_indices is not None:
            assert self._context_range is not None and isinstance(self._context_range[0], (int, long, float))
            next_states = tf.concat([tf.random_uniform(shape=next_states[:, :1].shape, minval=self._context_range[0], maxval=self._context_range[1], dtype=next_states.dtype) if i in self._override_indices else next_states[:, i:i + 1] for i in range(self._context_spec.shape.as_list()[0])], axis=1)
        if self._state_indices is not None:
            next_states = tf.concat([next_states[:, i:i + 1] for i in range(self._context_spec.shape.as_list()[0])], axis=1)
        self._validate_contexts(next_states)
        return (next_states, next_states)

@gin.configurable
class TimeSampler(BaseSampler):
    """Time Sampler."""

    def __init__(self, minval=0, maxval=1, timestep=-1, *args, **kwargs):
        if False:
            return 10
        'Construct sampler.\n\n    Args:\n      minval: Min value integer.\n      maxval: Max value integer.\n      timestep: Time step between states and next_states.\n      *args: arguments.\n      **kwargs: keyword arguments.\n    '
        super(TimeSampler, self).__init__(*args, **kwargs)
        assert self._context_spec.shape.as_list() == [1]
        self._minval = minval
        self._maxval = maxval
        self._timestep = timestep

    def __call__(self, batch_size, **kwargs):
        if False:
            return 10
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        if self._maxval == self._minval:
            contexts = tf.constant(self._maxval, shape=[batch_size, 1], dtype=tf.int32)
        else:
            contexts = tf.random_uniform(shape=[batch_size, 1], dtype=tf.int32, maxval=self._maxval, minval=self._minval)
        next_contexts = tf.maximum(contexts + self._timestep, 0)
        return (tf.cast(contexts, dtype=self._context_spec.dtype), tf.cast(next_contexts, dtype=self._context_spec.dtype))

@gin.configurable
class ConstantSampler(BaseSampler):
    """Constant sampler."""

    def __init__(self, value=None, *args, **kwargs):
        if False:
            print('Hello World!')
        'Construct sampler.\n\n    Args:\n      value: A list or Numpy array for values of the constant.\n      *args: arguments.\n      **kwargs: keyword arguments.\n    '
        super(ConstantSampler, self).__init__(*args, **kwargs)
        self._value = value

    def __call__(self, batch_size, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        spec = self._context_spec
        value_ = tf.constant(self._value, shape=spec.shape, dtype=spec.dtype)
        values = tf.tile(tf.expand_dims(value_, 0), (batch_size,) + (1,) * spec.shape.ndims)
        self._validate_contexts(values)
        return (values, values)

@gin.configurable
class DirectionSampler(RandomSampler):
    """Direction sampler."""

    def __call__(self, batch_size, **kwargs):
        if False:
            while True:
                i = 10
        'Sample a batch of context.\n\n    Args:\n      batch_size: Batch size.\n    Returns:\n      Two [batch_size, num_context_dims] tensors.\n    '
        spec = self._context_spec
        context_range = self._context_range
        if isinstance(context_range[0], (int, float)):
            contexts = tf.random_uniform(shape=[batch_size] + spec.shape.as_list(), minval=context_range[0], maxval=context_range[1], dtype=spec.dtype)
        elif isinstance(context_range[0], (list, tuple, np.ndarray)):
            assert len(spec.shape.as_list()) == 1
            assert spec.shape.as_list()[0] == len(context_range[0])
            assert spec.shape.as_list()[0] == len(context_range[1])
            contexts = tf.concat([tf.random_uniform(shape=[batch_size, 1] + spec.shape.as_list()[1:], minval=context_range[0][i], maxval=context_range[1][i], dtype=spec.dtype) for i in range(spec.shape.as_list()[0])], axis=1)
        else:
            raise NotImplementedError(context_range)
        self._validate_contexts(contexts)
        if 'sampler_fn' in kwargs:
            other_contexts = kwargs['sampler_fn']()
        else:
            other_contexts = contexts
        (state, next_state) = (kwargs['state'], kwargs['next_state'])
        if state is not None and next_state is not None:
            my_context_range = (np.array(context_range[1]) - np.array(context_range[0])) / 2 * np.ones(spec.shape.as_list())
            contexts = tf.concat([0.1 * my_context_range[:self._k] * tf.random_normal(tf.shape(state[:, :self._k]), dtype=state.dtype) + tf.random_shuffle(state[:, :self._k]) - state[:, :self._k], other_contexts[:, self._k:]], 1)
            next_contexts = tf.concat([state[:, :self._k] + contexts[:, :self._k] - next_state[:, :self._k], other_contexts[:, self._k:]], 1)
            next_contexts = contexts
        else:
            next_contexts = contexts
        return (tf.stop_gradient(contexts), tf.stop_gradient(next_contexts))