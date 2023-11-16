"""Helpers for Network Regularizers that are bilinear in their inputs/outputs.

Examples: The number of FLOPs and the number weights of a convolution are both
a bilinear expression in the number of its inputs and outputs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from morph_net.framework import generic_regularizers
_CONV2D_OPS = ('Conv2D', 'Conv2DBackpropInput', 'DepthwiseConv2dNative')
_SUPPORTED_OPS = _CONV2D_OPS + ('MatMul',)

def _raise_if_not_supported(op):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(op, tf.Operation):
        raise ValueError('conv_op must be a tf.Operation, not %s' % type(op))
    if op.type not in _SUPPORTED_OPS:
        raise ValueError('conv_op must be a Conv2D or a MatMul, not %s' % op.type)

def _get_conv_filter_size(conv_op):
    if False:
        while True:
            i = 10
    assert conv_op.type in _CONV2D_OPS
    conv_weights = conv_op.inputs[1]
    filter_shape = conv_weights.shape.as_list()[:2]
    return filter_shape[0] * filter_shape[1]

def flop_coeff(op):
    if False:
        return 10
    "Computes the coefficient of number of flops associated with a convolution.\n\n  The FLOPs cost of a convolution is given by C * output_depth * input_depth,\n  where C = 2 * output_width * output_height * filter_size. The 2 is because we\n  have one multiplication and one addition for each convolution weight and\n  pixel. This function returns C.\n\n  Args:\n    op: A tf.Operation of type 'Conv2D' or 'MatMul'.\n\n  Returns:\n    A float, the coefficient that when multiplied by the input depth and by the\n    output depth gives the number of flops needed to compute the convolution.\n\n  Raises:\n    ValueError: conv_op is not a tf.Operation of type Conv2D.\n  "
    _raise_if_not_supported(op)
    if op.type in _CONV2D_OPS:
        if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
            shape = op.outputs[0].shape.as_list()
        else:
            shape = _get_input(op).shape.as_list()
        size = shape[1] * shape[2]
        return 2.0 * size * _get_conv_filter_size(op)
    else:
        return 2.0

def num_weights_coeff(op):
    if False:
        while True:
            i = 10
    "The number of weights of a conv is C * output_depth * input_depth. Finds C.\n\n  Args:\n    op: A tf.Operation of type 'Conv2D' or 'MatMul'\n\n  Returns:\n    A float, the coefficient that when multiplied by the input depth and by the\n    output depth gives the number of flops needed to compute the convolution.\n\n  Raises:\n    ValueError: conv_op is not a tf.Operation of type Conv2D.\n  "
    _raise_if_not_supported(op)
    return _get_conv_filter_size(op) if op.type in _CONV2D_OPS else 1.0

class BilinearNetworkRegularizer(generic_regularizers.NetworkRegularizer):
    """A NetworkRegularizer with bilinear cost and loss.

  Can be used for FLOPs regularization or for model size regularization.
  """

    def __init__(self, opreg_manager, coeff_func):
        if False:
            while True:
                i = 10
        'Creates an instance.\n\n    Args:\n      opreg_manager: An OpRegularizerManager object that will be used to query\n        OpRegularizers of the various ops in the graph.\n      coeff_func: A callable that receives a tf.Operation of type Conv2D and\n        returns a bilinear coefficient of its cost. Examples:\n        - Use conv_flop_coeff for a FLOP regularizer.\n        - Use conv_num_weights_coeff for a number-of-weights regularizer.\n    '
        self._opreg_manager = opreg_manager
        self._coeff_func = coeff_func

    def _get_cost_or_regularization_term(self, is_regularization, ops=None):
        if False:
            for i in range(10):
                print('nop')
        total = 0.0
        if not ops:
            ops = self._opreg_manager.ops
        for op in ops:
            if op.type not in _SUPPORTED_OPS:
                continue
            input_op = _get_input(op).op
            input_op_reg = self._opreg_manager.get_regularizer(input_op)
            output_op_reg = self._opreg_manager.get_regularizer(op)
            coeff = self._coeff_func(op)
            num_alive_inputs = _count_alive(input_op, input_op_reg)
            num_alive_outputs = _count_alive(op, output_op_reg)
            if op.type == 'DepthwiseConv2dNative':
                if is_regularization:
                    reg_inputs = _sum_of_reg_vector(input_op_reg)
                    reg_outputs = _sum_of_reg_vector(output_op_reg)
                    total += coeff * (reg_inputs + reg_outputs)
                else:
                    total += coeff * num_alive_outputs
            elif is_regularization:
                reg_inputs = _sum_of_reg_vector(input_op_reg)
                reg_outputs = _sum_of_reg_vector(output_op_reg)
                total += coeff * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
            else:
                total += coeff * num_alive_inputs * num_alive_outputs
        return total

    def get_cost(self, ops=None):
        if False:
            for i in range(10):
                print('nop')
        return self._get_cost_or_regularization_term(False, ops)

    def get_regularization_term(self, ops=None):
        if False:
            for i in range(10):
                print('nop')
        return self._get_cost_or_regularization_term(True, ops)

def _get_input(op):
    if False:
        print('Hello World!')
    'Returns the input to that op that represents the activations.\n\n  (as opposed to e.g. weights.)\n\n  Args:\n    op: A tf.Operation object with type in _SUPPORTED_OPS.\n\n  Returns:\n    A tf.Tensor representing the input activations.\n\n  Raises:\n    ValueError: MatMul is used with transposition (unsupported).\n  '
    assert op.type in _SUPPORTED_OPS, 'Op type %s is not supported.' % op.type
    if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
        return op.inputs[0]
    if op.type == 'Conv2DBackpropInput':
        return op.inputs[2]
    if op.type == 'MatMul':
        if op.get_attr('transpose_a') or op.get_attr('transpose_b'):
            raise ValueError('MatMul with transposition is not yet supported.')
        return op.inputs[0]

def _count_alive(op, opreg):
    if False:
        return 10
    if opreg:
        return tf.reduce_sum(tf.cast(opreg.alive_vector, tf.float32))
    else:
        return float(op.outputs[0].shape.as_list()[-1])

def _sum_of_reg_vector(opreg):
    if False:
        i = 10
        return i + 15
    if opreg:
        return tf.reduce_sum(opreg.regularization_vector)
    else:
        return 0.0