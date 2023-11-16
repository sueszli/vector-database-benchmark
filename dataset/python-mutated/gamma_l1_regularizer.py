"""An OpRegularizer that applies L1 regularization on batch-norm gammas."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from morph_net.framework import generic_regularizers
from morph_net.op_regularizers import gamma_mapper

class GammaL1Regularizer(generic_regularizers.OpRegularizer):
    """An OpRegularizer that L1-regularizes batch-norm gamma."""

    def __init__(self, gamma, gamma_threshold):
        if False:
            for i in range(10):
                print('nop')
        "Creates an instance.\n\n    Args:\n      gamma: a tf.Tensor of rank 1 with the gammas.\n      gamma_threshold: A float scalar, the threshold above which a gamma is\n        considered 'alive'.\n    "
        self._gamma = gamma
        self._gamma_threshold = gamma_threshold
        abs_gamma = tf.abs(gamma)
        self._alive_vector = abs_gamma > gamma_threshold
        self._regularization_vector = abs_gamma

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

class GammaL1RegularizerFactory(object):
    """A class for creating a GammaL1Regularizer for convolutions."""

    def __init__(self, gamma_threshold):
        if False:
            for i in range(10):
                print('nop')
        "Creates an instance.\n\n    Args:\n      gamma_threshold: A float scalar, will be used as a 'gamma_threshold' for\n        all the GammaL1Regularizer-s created by this class.\n    "
        self._gamma_conv_mapper = gamma_mapper.ConvGammaMapperByName()
        self._gamma_threshold = gamma_threshold

    def create_regularizer(self, op, opreg_manager):
        if False:
            for i in range(10):
                print('nop')
        "Creates a GammaL1Regularizer for `op`.\n\n    Args:\n      op: A tf.Operation of type 'Conv2D' or 'DepthwiseConv2dNative'.\n      opreg_manager: An OpRegularizerManager object that will host the created\n        OpRegularizer object.\n\n    Returns:\n      a GammaL1Regularizer that corresponds to `op`.\n\n    Raises:\n      ValueError: If `op` does not have a Gamma that corresponds to it.\n    "
        gamma = self._gamma_conv_mapper.get_gamma(op)
        if gamma is None:
            regularizer = None
        else:
            regularizer = GammaL1Regularizer(gamma, self._gamma_threshold)
        if op.type == 'DepthwiseConv2dNative':
            regularizer = _group_depthwise_conv_regularizer(op, regularizer, opreg_manager)
        return regularizer

def _group_depthwise_conv_regularizer(op, regularizer, opreg_manager):
    if False:
        return 10
    'Groups the regularizer of a depthwise convolution if needed.'
    input_reg = opreg_manager.get_regularizer(op.inputs[0].op)
    if input_reg is None:
        return None
    if op.inputs[0].shape.as_list()[-1] != op.outputs[0].shape.as_list()[-1]:
        return None
    if regularizer is None:
        return input_reg
    else:
        return opreg_manager.group_and_replace_regularizers([regularizer, input_reg])