"""A circular buffer where each element is a list of tensors.

Each element of the buffer is a list of tensors. An example use case is a replay
buffer in reinforcement learning, where each element is a list of tensors
representing the state, action, reward etc.

New elements are added sequentially, and once the buffer is full, we
start overwriting them in a circular fashion. Reading does not remove any
elements, only adding new elements does.
"""
import collections
import numpy as np
import tensorflow as tf
import gin.tf

@gin.configurable
class CircularBuffer(object):
    """A circular buffer where each element is a list of tensors."""

    def __init__(self, buffer_size=1000, scope='replay_buffer'):
        if False:
            print('Hello World!')
        'Circular buffer of list of tensors.\n\n    Args:\n      buffer_size: (integer) maximum number of tensor lists the buffer can hold.\n      scope: (string) variable scope for creating the variables.\n    '
        self._buffer_size = np.int64(buffer_size)
        self._scope = scope
        self._tensors = collections.OrderedDict()
        with tf.variable_scope(self._scope):
            self._num_adds = tf.Variable(0, dtype=tf.int64, name='num_adds')
        self._num_adds_cs = tf.CriticalSection(name='num_adds')

    @property
    def buffer_size(self):
        if False:
            while True:
                i = 10
        return self._buffer_size

    @property
    def scope(self):
        if False:
            while True:
                i = 10
        return self._scope

    @property
    def num_adds(self):
        if False:
            print('Hello World!')
        return self._num_adds

    def _create_variables(self, tensors):
        if False:
            i = 10
            return i + 15
        with tf.variable_scope(self._scope):
            for name in tensors.keys():
                tensor = tensors[name]
                self._tensors[name] = tf.get_variable(name='BufferVariable_' + name, shape=[self._buffer_size] + tensor.get_shape().as_list(), dtype=tensor.dtype, trainable=False)

    def _validate(self, tensors):
        if False:
            i = 10
            return i + 15
        'Validate shapes of tensors.'
        if len(tensors) != len(self._tensors):
            raise ValueError('Expected tensors to have %d elements. Received %d instead.' % (len(self._tensors), len(tensors)))
        if self._tensors.keys() != tensors.keys():
            raise ValueError('The keys of tensors should be the always the same.Received %s instead %s.' % (tensors.keys(), self._tensors.keys()))
        for (name, tensor) in tensors.items():
            if tensor.get_shape().as_list() != self._tensors[name].get_shape().as_list()[1:]:
                raise ValueError('Tensor %s has incorrect shape.' % name)
            if not tensor.dtype.is_compatible_with(self._tensors[name].dtype):
                raise ValueError('Tensor %s has incorrect data type. Expected %s, received %s' % (name, self._tensors[name].read_value().dtype, tensor.dtype))

    def add(self, tensors):
        if False:
            while True:
                i = 10
        "Adds an element (list/tuple/dict of tensors) to the buffer.\n\n    Args:\n      tensors: (list/tuple/dict of tensors) to be added to the buffer.\n    Returns:\n      An add operation that adds the input `tensors` to the buffer. Similar to\n        an enqueue_op.\n    Raises:\n      ValueError: If the shapes and data types of input `tensors' are not the\n        same across calls to the add function.\n    "
        return self.maybe_add(tensors, True)

    def maybe_add(self, tensors, condition):
        if False:
            for i in range(10):
                print('nop')
        "Adds an element (tensors) to the buffer based on the condition..\n\n    Args:\n      tensors: (list/tuple of tensors) to be added to the buffer.\n      condition: A boolean Tensor controlling whether the tensors would be added\n        to the buffer or not.\n    Returns:\n      An add operation that adds the input `tensors` to the buffer. Similar to\n        an maybe_enqueue_op.\n    Raises:\n      ValueError: If the shapes and data types of input `tensors' are not the\n        same across calls to the add function.\n    "
        if not isinstance(tensors, dict):
            names = [str(i) for i in range(len(tensors))]
            tensors = collections.OrderedDict(zip(names, tensors))
        if not isinstance(tensors, collections.OrderedDict):
            tensors = collections.OrderedDict(sorted(tensors.items(), key=lambda t: t[0]))
        if not self._tensors:
            self._create_variables(tensors)
        else:
            self._validate(tensors)

        def _increment_num_adds():
            if False:
                print('Hello World!')
            return self._num_adds.assign_add(1) + 0

        def _add():
            if False:
                while True:
                    i = 10
            num_adds_inc = self._num_adds_cs.execute(_increment_num_adds)
            current_pos = tf.mod(num_adds_inc - 1, self._buffer_size)
            update_ops = []
            for name in self._tensors.keys():
                update_ops.append(tf.scatter_update(self._tensors[name], current_pos, tensors[name]))
            return tf.group(*update_ops)
        return tf.contrib.framework.smart_cond(condition, _add, tf.no_op)

    def get_random_batch(self, batch_size, keys=None, num_steps=1):
        if False:
            for i in range(10):
                print('nop')
        'Samples a batch of tensors from the buffer with replacement.\n\n    Args:\n      batch_size: (integer) number of elements to sample.\n      keys: List of keys of tensors to retrieve. If None retrieve all.\n      num_steps: (integer) length of trajectories to return. If > 1 will return\n        a list of lists, where each internal list represents a trajectory of\n        length num_steps.\n    Returns:\n      A list of tensors, where each element in the list is a batch sampled from\n        one of the tensors in the buffer.\n    Raises:\n      ValueError: If get_random_batch is called before calling the add function.\n      tf.errors.InvalidArgumentError: If this operation is executed before any\n        items are added to the buffer.\n    '
        if not self._tensors:
            raise ValueError('The add function must be called before get_random_batch.')
        if keys is None:
            keys = self._tensors.keys()
        latest_start_index = self.get_num_adds() - num_steps + 1
        empty_buffer_assert = tf.Assert(tf.greater(latest_start_index, 0), ['Not enough elements have been added to the buffer.'])
        with tf.control_dependencies([empty_buffer_assert]):
            max_index = tf.minimum(self._buffer_size, latest_start_index)
            indices = tf.random_uniform([batch_size], minval=0, maxval=max_index, dtype=tf.int64)
            if num_steps == 1:
                return self.gather(indices, keys)
            else:
                return self.gather_nstep(num_steps, indices, keys)

    def gather(self, indices, keys=None):
        if False:
            print('Hello World!')
        'Returns elements at the specified indices from the buffer.\n\n    Args:\n      indices: (list of integers or rank 1 int Tensor) indices in the buffer to\n        retrieve elements from.\n      keys: List of keys of tensors to retrieve. If None retrieve all.\n    Returns:\n      A list of tensors, where each element in the list is obtained by indexing\n        one of the tensors in the buffer.\n    Raises:\n      ValueError: If gather is called before calling the add function.\n      tf.errors.InvalidArgumentError: If indices are bigger than the number of\n        items in the buffer.\n    '
        if not self._tensors:
            raise ValueError('The add function must be called before calling gather.')
        if keys is None:
            keys = self._tensors.keys()
        with tf.name_scope('Gather'):
            index_bound_assert = tf.Assert(tf.less(tf.to_int64(tf.reduce_max(indices)), tf.minimum(self.get_num_adds(), self._buffer_size)), ['Index out of bounds.'])
            with tf.control_dependencies([index_bound_assert]):
                indices = tf.convert_to_tensor(indices)
            batch = []
            for key in keys:
                batch.append(tf.gather(self._tensors[key], indices, name=key))
            return batch

    def gather_nstep(self, num_steps, indices, keys=None):
        if False:
            while True:
                i = 10
        'Returns elements at the specified indices from the buffer.\n\n    Args:\n      num_steps: (integer) length of trajectories to return.\n      indices: (list of rank num_steps int Tensor) indices in the buffer to\n        retrieve elements from for multiple trajectories. Each Tensor in the\n        list represents the indices for a trajectory.\n      keys: List of keys of tensors to retrieve. If None retrieve all.\n    Returns:\n      A list of list-of-tensors, where each element in the list is obtained by\n        indexing one of the tensors in the buffer.\n    Raises:\n      ValueError: If gather is called before calling the add function.\n      tf.errors.InvalidArgumentError: If indices are bigger than the number of\n        items in the buffer.\n    '
        if not self._tensors:
            raise ValueError('The add function must be called before calling gather.')
        if keys is None:
            keys = self._tensors.keys()
        with tf.name_scope('Gather'):
            index_bound_assert = tf.Assert(tf.less_equal(tf.to_int64(tf.reduce_max(indices) + num_steps), self.get_num_adds()), ['Trajectory indices go out of bounds.'])
            with tf.control_dependencies([index_bound_assert]):
                indices = tf.map_fn(lambda x: tf.mod(tf.range(x, x + num_steps), self._buffer_size), indices, dtype=tf.int64)
            batch = []
            for key in keys:

                def SampleTrajectories(trajectory_indices, key=key, num_steps=num_steps):
                    if False:
                        for i in range(10):
                            print('nop')
                    trajectory_indices.set_shape([num_steps])
                    return tf.gather(self._tensors[key], trajectory_indices, name=key)
                batch.append(tf.map_fn(SampleTrajectories, indices, dtype=self._tensors[key].dtype))
            return batch

    def get_position(self):
        if False:
            i = 10
            return i + 15
        'Returns the position at which the last element was added.\n\n    Returns:\n      An int tensor representing the index at which the last element was added\n        to the buffer or -1 if no elements were added.\n    '
        return tf.cond(self.get_num_adds() < 1, lambda : self.get_num_adds() - 1, lambda : tf.mod(self.get_num_adds() - 1, self._buffer_size))

    def get_num_adds(self):
        if False:
            while True:
                i = 10
        'Returns the number of additions to the buffer.\n\n    Returns:\n      An int tensor representing the number of elements that were added.\n    '

        def num_adds():
            if False:
                while True:
                    i = 10
            return self._num_adds.value()
        return self._num_adds_cs.execute(num_adds)

    def get_num_tensors(self):
        if False:
            print('Hello World!')
        'Returns the number of tensors (slots) in the buffer.'
        return len(self._tensors)