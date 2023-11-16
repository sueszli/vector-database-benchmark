"""A NetworkRegularizer that targets the number of weights in the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import op_regularizer_manager
from morph_net.network_regularizers import bilinear_cost_utils
from morph_net.op_regularizers import gamma_l1_regularizer

class GammaModelSizeRegularizer(bilinear_cost_utils.BilinearNetworkRegularizer):
    """A NetworkRegularizer that targets model size using Gamma L1 OpReg."""

    def __init__(self, ops, gamma_threshold):
        if False:
            return 10
        gamma_l1_reg_factory = gamma_l1_regularizer.GammaL1RegularizerFactory(gamma_threshold)
        opreg_manager = op_regularizer_manager.OpRegularizerManager(ops, {'Conv2D': gamma_l1_reg_factory.create_regularizer, 'DepthwiseConv2dNative': gamma_l1_reg_factory.create_regularizer})
        super(GammaModelSizeRegularizer, self).__init__(opreg_manager, bilinear_cost_utils.num_weights_coeff)