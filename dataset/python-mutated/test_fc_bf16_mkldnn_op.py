import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core

def fully_connected_naive(input, weights, bias_data):
    if False:
        i = 10
        return i + 15
    result = np.dot(input, weights) + bias_data
    return result

class MatrixGenerate:

    def __init__(self, mb, ic, oc, h, w):
        if False:
            print('Hello World!')
        self.input = np.random.random((mb, ic * h * w)).astype(np.float32)
        self.weights = np.random.random((ic * h * w, oc)).astype(np.float32)

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestFcBf16MklDNNOp(OpTest):

    def generate_data(self):
        if False:
            i = 10
            return i + 15
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype('float32')

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fc'
        self.use_mkldnn = True
        self.mkldnn_data_type = 'bfloat16'
        self.force_fp32_output = False
        self.generate_data()
        self.output = fully_connected_naive(self.matrix.input, self.matrix.weights, self.bias)
        if not self.force_fp32_output:
            self.output = convert_float_to_uint16(self.output)
        self.inputs = {'Input': convert_float_to_uint16(self.matrix.input), 'W': self.matrix.weights, 'Bias': self.bias}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'force_fp32_output': self.force_fp32_output}
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_no_weight(self):
        if False:
            return 10
        pass

class TestFCMKLDNNOp1(TestFcBf16MklDNNOp):

    def generate_data(self):
        if False:
            return 10
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype(np.float32)
if __name__ == '__main__':
    enable_static()
    unittest.main()