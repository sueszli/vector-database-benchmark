"""Regularizers that group other regularizers for residual connections.

An Elementwise operation between two tensors (addition, multiplication, maximum
etc) imposes a constraint of equality of the shapes of the constituents. For
example, if A, B are convolutions, and another op in the network
receives A + B as input, it means that the i-th output of A is tied to the i-th
output of B. Only if the i-th output was regularized away by the reguarizer in
both A and B can we discard the i-th activation in both.

Therefore we group the i-th output of A and the i-th output of B in a group
LASSO, a group for each i. The grouping methods can vary, and this file offers
several variants.

Residual connections, in ResNet or in RNNs, are examples where this kind of
grouping is needed.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from morph_net.framework import generic_regularizers
DEFAULT_THRESHOLD = 0.01

class MaxGroupingRegularizer(generic_regularizers.OpRegularizer):
    """A regularizer that groups others by taking their maximum."""

    def __init__(self, regularizers_to_group):
        if False:
            while True:
                i = 10
        'Creates an instance.\n\n    Args:\n      regularizers_to_group: A list of generic_regularizers.OpRegularizer\n        objects.Their regularization_vector (alive_vector) are expected to be of\n        the same length.\n\n    Raises:\n      ValueError: regularizers_to_group is not of length 2 (TODO:\n        support arbitrary length if needed.\n    '
        _raise_if_length_is_not2(regularizers_to_group)
        self._regularization_vector = tf.maximum(regularizers_to_group[0].regularization_vector, regularizers_to_group[1].regularization_vector)
        self._alive_vector = tf.logical_or(regularizers_to_group[0].alive_vector, regularizers_to_group[1].alive_vector)

    @property
    def regularization_vector(self):
        if False:
            print('Hello World!')
        return self._regularization_vector

    @property
    def alive_vector(self):
        if False:
            i = 10
            return i + 15
        return self._alive_vector

class L2GroupingRegularizer(generic_regularizers.OpRegularizer):
    """A regularizer that groups others by taking their L2 norm.

  R_j = sqrt((\\sum_i r_{ij}^2))

  Where r_i is the i-th regularization vector, r_{ij} is its j-th element, and
  R_j is the j-th element of the resulting regularization vector.
  """

    def __init__(self, regularizers_to_group, threshold=DEFAULT_THRESHOLD):
        if False:
            while True:
                i = 10
        'Creates an instance.\n\n    Args:\n      regularizers_to_group: A list of generic_regularizers.OpRegularizer\n        objects.Their regularization_vector (alive_vector) are expected to be of\n        the same length.\n      threshold: A float. An group of activations will be considered alive if\n        its L2 norm is greater than `threshold`.\n\n    Raises:\n      ValueError: regularizers_to_group is not of length 2 (TODO:\n        support arbitrary length if needed.\n    '
        _raise_if_length_is_not2(regularizers_to_group)
        self._regularization_vector = tf.sqrt(lazy_square(regularizers_to_group[0].regularization_vector) + lazy_square(regularizers_to_group[1].regularization_vector))
        self._alive_vector = self._regularization_vector > threshold

    @property
    def regularization_vector(self):
        if False:
            return 10
        return self._regularization_vector

    @property
    def alive_vector(self):
        if False:
            i = 10
            return i + 15
        return self._alive_vector

def _raise_if_length_is_not2(regularizers_to_group):
    if False:
        return 10
    if len(regularizers_to_group) != 2:
        raise ValueError('Currently only groups of size 2 are supported.')

def lazy_square(tensor):
    if False:
        return 10
    'Computes the square of a tensor in a lazy way.\n\n  This function is lazy in the following sense, for:\n    tensor = tf.sqrt(input)\n  will return input (and not tf.square(tensor)).\n\n  Args:\n    tensor: A `Tensor` of floats to compute the square of.\n\n  Returns:\n    The squre of the input tensor.\n  '
    if tensor.op.type == 'Sqrt':
        return tensor.op.inputs[0]
    else:
        return tf.square(tensor)