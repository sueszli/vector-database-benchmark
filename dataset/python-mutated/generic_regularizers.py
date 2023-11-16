"""Interface for MorphNet regularizers framework.

A subclasses of Regularizer represent a regularizer that targets a certain
quantity: Number of flops, model size, number of activations etc. The
Regularizer interface has two methods:

1. `get_regularization_term`, which returns a regularization term that should be
   included in the total loss to target the quantity.

2. `get_cost`, the quantity itself (for example, the number of flops). This is
   useful for display in TensorBoard, and later, to to provide feedback for
   automatically tuning the coefficient that multplies the regularization term,
   until the cost reaches (or goes below) its target value.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

class OpRegularizer(object):
    """An interface for Op Regularizers.

  An OpRegularizer object corresponds to a tf.Operation, and provides
  a regularizer for the output of the op (we assume that the op has one output
  of interest in the context of MorphNet).
  """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def regularization_vector(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a vector of floats, with regularizers.\n\n    The length of the vector is the number of "output activations" (call them\n    neurons, nodes, filters etc) of the op. For a convolutional network, it\'s\n    the number of filters (aka "depth"). For a fully-connected layer, it\'s\n    usually the second (and last) dimension - assuming the first one is the\n    batch size.\n    '
        pass

    @abc.abstractproperty
    def alive_vector(self):
        if False:
            i = 10
            return i + 15
        'Returns a vector of booleans, indicating which activations are alive.\n\n    (call them activations, neurons, nodes, filters etc). This vector is of the\n    same length as the regularization_vector.\n    '
        pass

class NetworkRegularizer(object):
    """An interface for Network Regularizers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_regularization_term(self, ops=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the regularization term.\n\n    Args:\n      ops: A list of tf.Operation objects. If specified, only the regularization\n        term associated with the ops in `ops` will be returned. Otherwise, all\n        relevant ops in the default TensorFlow graph will be included.\n\n    Returns:\n      A tf.Tensor scalar of floating point type that evaluates to the\n      regularization term (that should be added to the total loss, with a\n      suitable coefficient)\n    '
        pass

    @abc.abstractmethod
    def get_cost(self, ops=None):
        if False:
            return 10
        'Calculates the cost targeted by the Regularizer.\n\n    Args:\n      ops: A list of tf.Operation objects. If specified, only the cost\n        pertaining to the ops in the `ops` will be returned. Otherwise, all\n        relevant ops in the default TensorFlow graph will be included.\n\n    Returns:\n      A tf.Tensor scalar that evaluates to the cost.\n    '
        pass

def dimensions_are_compatible(op_regularizer):
    if False:
        print('Hello World!')
    "Checks if op_regularizer's alive_vector matches regularization_vector."
    return op_regularizer.alive_vector.shape.with_rank(1).dims[0].is_compatible_with(op_regularizer.regularization_vector.shape.with_rank(1).dims[0])