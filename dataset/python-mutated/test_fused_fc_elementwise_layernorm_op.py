import unittest
import numpy as np
from op_test import OpTest
from test_fc_op import MatrixGenerate, fc_refer
from test_layer_norm_op import _reference_layer_norm_naive
from paddle.base import core
np.random.random(123)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'Paddle core is not compiled with CUDA')
class TestFusedFCElementwiseLayerNormOp(OpTest):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3, 2)
        self.y_shape = [1, 15]
        self.begin_norm_axis = 1

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fused_fc_elementwise_layernorm'
        self.config()
        epsilon = 1e-05
        fc_out = fc_refer(self.matrix, True, True)
        y = np.random.random_sample(self.y_shape).astype(np.float32)
        add_out = fc_out + y
        scale_shape = [np.prod(self.y_shape[self.begin_norm_axis:])]
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias_1 = np.random.random_sample(scale_shape).astype(np.float32)
        (out, mean, variance) = _reference_layer_norm_naive(add_out, scale, bias_1, epsilon, self.begin_norm_axis)
        self.inputs = {'X': self.matrix.input, 'W': self.matrix.weights, 'Bias0': self.matrix.bias, 'Y': y, 'Scale': scale, 'Bias1': bias_1}
        self.attrs = {'activation_type': 'relu', 'epsilon': epsilon, 'begin_norm_axis': self.begin_norm_axis}
        self.outputs = {'Out': out, 'Mean': mean, 'Variance': variance}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=0.002, check_dygraph=False)

class TestFusedFCElementwiseLayerNormOp2(TestFusedFCElementwiseLayerNormOp):

    def config(self):
        if False:
            return 10
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)
        self.y_shape = [4, 6]
        self.begin_norm_axis = 1
if __name__ == '__main__':
    unittest.main()