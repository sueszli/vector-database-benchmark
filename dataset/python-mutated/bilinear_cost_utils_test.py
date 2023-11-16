"""Tests for compute_cost_estimator.

Note that BilinearNetworkRegularizer is not tested here - its specific
instantiation is tested in flop_regularizer_test.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from morph_net.network_regularizers import bilinear_cost_utils
layers = tf.contrib.layers

def _flops(op):
    if False:
        print('Hello World!')
    'Get the number of flops of a convolution, from the ops stats registry.\n\n  Args:\n    op: A tf.Operation object.\n\n  Returns:\n    The number os flops needed to evaluate conv_op.\n  '
    return ops.get_stats_for_node_def(tf.get_default_graph(), op.node_def, 'flops').value

def _output_depth(conv_op):
    if False:
        while True:
            i = 10
    return conv_op.outputs[0].shape.as_list()[-1]

def _input_depth(conv_op):
    if False:
        return 10
    conv_weights = conv_op.inputs[1]
    return conv_weights.shape.as_list()[2]

class BilinearCostUtilTest(tf.test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        tf.reset_default_graph()
        image = tf.constant(0.0, shape=[1, 11, 13, 17])
        net = layers.conv2d(image, 19, [7, 5], stride=2, padding='SAME', scope='conv1')
        layers.conv2d_transpose(image, 29, [7, 5], stride=2, padding='SAME', scope='convt2')
        net = tf.reduce_mean(net, axis=(1, 2))
        layers.fully_connected(net, 23, scope='FC')
        net = layers.conv2d(image, 10, [7, 5], stride=2, padding='SAME', scope='conv2')
        layers.separable_conv2d(net, None, [3, 2], depth_multiplier=1, padding='SAME', scope='dw1')
        self.conv_op = tf.get_default_graph().get_operation_by_name('conv1/Conv2D')
        self.convt_op = tf.get_default_graph().get_operation_by_name('convt2/conv2d_transpose')
        self.matmul_op = tf.get_default_graph().get_operation_by_name('FC/MatMul')
        self.dw_op = tf.get_default_graph().get_operation_by_name('dw1/depthwise')

    def assertNearRelatively(self, expected, actual):
        if False:
            i = 10
            return i + 15
        self.assertNear(expected, actual, expected * 1e-06)

    def testConvFlopsCoeff(self):
        if False:
            print('Hello World!')
        expected_coeff = _flops(self.conv_op) / (17.0 * 19.0)
        actual_coeff = bilinear_cost_utils.flop_coeff(self.conv_op)
        self.assertNearRelatively(expected_coeff, actual_coeff)

    def testConvTransposeFlopsCoeff(self):
        if False:
            for i in range(10):
                print('nop')
        expected_coeff = _flops(self.convt_op) / (17.0 * 29.0)
        actual_coeff = bilinear_cost_utils.flop_coeff(self.convt_op)
        self.assertNearRelatively(expected_coeff, actual_coeff)

    def testFcFlopsCoeff(self):
        if False:
            return 10
        expected_coeff = _flops(self.matmul_op) / (19.0 * 23.0)
        actual_coeff = bilinear_cost_utils.flop_coeff(self.matmul_op)
        self.assertNearRelatively(expected_coeff, actual_coeff)

    def testConvNumWeightsCoeff(self):
        if False:
            return 10
        actual_coeff = bilinear_cost_utils.num_weights_coeff(self.conv_op)
        self.assertNearRelatively(35, actual_coeff)

    def testFcNumWeightsCoeff(self):
        if False:
            for i in range(10):
                print('nop')
        actual_coeff = bilinear_cost_utils.num_weights_coeff(self.matmul_op)
        self.assertNearRelatively(1.0, actual_coeff)

    def testDepthwiseConvFlopsCoeff(self):
        if False:
            while True:
                i = 10
        expected_coeff = _flops(self.dw_op) / 10.0
        actual_coeff = bilinear_cost_utils.flop_coeff(self.dw_op)
        self.assertNearRelatively(expected_coeff, actual_coeff)
if __name__ == '__main__':
    tf.test.main()