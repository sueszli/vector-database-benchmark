"""Classes for mapping convolutions to their batch-norm gammas."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import tensorflow as tf
from morph_net.framework import op_regularizer_manager

class GenericConvGammaMapper(object):
    """An interface for mapping convolutions to their batch-norm gammas."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_gamma(self, conv_op):
        if False:
            return 10
        'Returns the BatchNorm gamma tensor associated with `conv_op`, or None.\n\n    Args:\n      conv_op: A tf.Operation of type Conv2D.\n\n    Returns:\n      A tf.Tensor containing the BatchNorm gamma associated with `conv_op`, or\n      None if `conv_op` has no BatchNorm gamma.\n\n    Raises:\n      ValueError: `conv_op` is not a tf.Operation of type `Conv2D`.\n      KeyError: `conv_op` is not in the graph that was used to construct `self`\n    '

    @abc.abstractproperty
    def all_conv_ops(self):
        if False:
            return 10
        'Return all Conv2D ops that were in the graph when `self` was created.'
        pass

def _get_existing_variable(name):
    if False:
        i = 10
        return i + 15
    "Fetches a variable by name (like tf.get_variable with reuse=True).\n\n  The reason why we can't simply use tf.get_variable with reuse=True is that\n  when variable partitioner is used, tf.get_variable requires knowing the shape\n  of the variable (even though it knows it and thus shouldn't require it). This\n  helper is a convenience function to solve this problem.\n\n  Args:\n    name: A string, the name of the variable.\n\n  Returns:\n    A tf.Tensor which is the result of convert_to_tensor of the variable, or\n    None if the variable does not exist.\n  "
    try:
        op = tf.get_default_graph().get_operation_by_name(name)
    except KeyError:
        return None
    try:
        shape = tf.TensorShape(op.get_attr('shape'))
    except ValueError:
        shape = op.outputs[0].shape
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        try:
            return tf.convert_to_tensor(tf.get_variable(name, shape=shape))
        except ValueError as e:
            if 'Variable %s does not exist' % name in str(e):
                return None
            else:
                raise e

class ConvGammaMapperByName(GenericConvGammaMapper):
    """Maps a convolution to its BatchNorm gamma.

  Assumes that the convolutions and their respective gammas conform to the
  naming convention of tf.contrib.layers: A convolution's name ends with
  `<BASE_NAME>/Conv2D`, and the respective batch-norm gamma ends with
  `<BASE_NAME>/BatchNorm/gamma`
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Constructs an instance. Builds mapping from Conv2D ops to their Gamma.'
        self._conv_to_gamma = {}
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for op in tf.get_default_graph().get_operations():
                if op.type != 'Conv2D' and op.type != 'DepthwiseConv2dNative':
                    continue
                base_name = op.name.rsplit('/', 1)[0]
                self._conv_to_gamma[op] = _get_existing_variable(base_name + '/BatchNorm/gamma')

    def get_gamma(self, conv_op):
        if False:
            i = 10
            return i + 15
        _raise_if_not_conv(conv_op)
        return self._conv_to_gamma[conv_op]

    @property
    def all_conv_ops(self):
        if False:
            return 10
        return self._conv_to_gamma.keys()

class ConvGammaMapperByConnectivity(GenericConvGammaMapper):
    """Maps a convolution to its BatchNorm gammas based on graph connectivity.

  Given a batch-norm gamma, propagates along the graph to find the convolutions
  that are batch-nomalized by this gamma. It can me more than one convolution
  that are normalized by the same batch-norm gamma in ResNet-s, where
  un-normalized convolutions are first summed and then their sum is normalized.
  The converse is also true - a single convolution can be connected (through
  residual connections) to multiple batch-norms.

  Only fused batch-norm is supported: there seems to be significant variability
  in the way non-fused batch-norm manifests in the tensorflow graph.
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Constructs an instance. Builds mapping from Conv2D ops to their Gamma.'
        self._conv_to_gamma = collections.defaultdict(set)
        for op in tf.get_default_graph().get_operations():
            if op.type != 'FusedBatchNorm':
                continue
            convs = _dfs(op)
            for conv in convs:
                if conv.type == 'Conv2D':
                    self._conv_to_gamma[conv].add(op.inputs[1])
        for op in tf.get_default_graph().get_operations():
            if op.type == 'Conv2D' and op not in self._conv_to_gamma:
                self._conv_to_gamma[op] = None

    def get_gamma(self, conv_op):
        if False:
            i = 10
            return i + 15
        _raise_if_not_conv(conv_op)
        if conv_op not in self._conv_to_gamma:
            raise KeyError
        gammas = self._conv_to_gamma[conv_op]
        if gammas and len(gammas) == 1:
            return list(gammas)[0]
        return gammas

    @property
    def all_conv_ops(self):
        if False:
            i = 10
            return i + 15
        return self._conv_to_gamma.keys()

def _dfs(op, visited=None):
    if False:
        print('Hello World!')
    'Perform DFS on a graph.\n\n  Args:\n    op: A tf.Operation, the root node for the DFS.\n    visited: A set, used in the recursion.\n\n  Returns:\n    A list of the tf.Operations of type Conv2D that were encountered.\n  '
    visited = visited or set()
    ret = []
    for child in op.inputs:
        if child.op in visited:
            return ret
        visited.add(child.op)
        if child.op.type not in op_regularizer_manager.NON_PASS_THROUGH_OPS:
            ret.extend(_dfs(child.op, visited))
        if child.op.type in ('Conv2D',):
            ret.append(child.op)
    return ret

def _raise_if_not_conv(op):
    if False:
        i = 10
        return i + 15
    if not isinstance(op, tf.Operation):
        raise ValueError('conv_op must be a tf.Operation, not %s' % type(op))
    if op.type != 'Conv2D' and op.type != 'DepthwiseConv2dNative':
        raise ValueError('conv_op must be a Conv2D or DepthwiseConv2dNative,not %s' % op.type)