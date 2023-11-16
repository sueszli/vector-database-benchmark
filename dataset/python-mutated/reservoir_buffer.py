"""Reservoir buffer implemented in Numpy.

See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
"""
import random
import numpy as np

class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

    def __init__(self, reservoir_buffer_capacity):
        if False:
            i = 10
            return i + 15
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        if False:
            for i in range(10):
                print('nop')
        'Potentially adds `element` to the reservoir buffer.\n\n    Args:\n      element: data to be added to the reservoir buffer.\n    '
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        if False:
            i = 10
            return i + 15
        'Returns `num_samples` uniformly sampled from the buffer.\n\n    Args:\n      num_samples: `int`, number of samples to draw.\n\n    Returns:\n      An iterable over `num_samples` random elements of the buffer.\n\n    Raises:\n      ValueError: If there are less than `num_samples` elements in the buffer\n    '
        if len(self._data) < num_samples:
            raise ValueError('{} elements could not be sampled from size {}'.format(num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        if False:
            print('Hello World!')
        self._data = []
        self._add_calls = 0

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._data)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._data)