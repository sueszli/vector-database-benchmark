import unittest
import numpy as np
from op_test import OpTest

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
        self.input = np.random.random((mb, ic * h * w)).astype('float32')
        self.weights = np.random.random((ic * h * w, oc)).astype('float32')

class TestFCMKLDNNOp(OpTest):

    def create_data(self):
        if False:
            i = 10
            return i + 15
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype('float32')

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fc'
        self._cpu_only = True
        self.use_mkldnn = True
        self.create_data()
        self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights, 'Bias': self.bias}
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': fully_connected_naive(self.matrix.input, self.matrix.weights, self.bias)}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_no_weight(self):
        if False:
            print('Hello World!')
        pass

class TestFCMKLDNNOp1(TestFCMKLDNNOp):

    def create_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype('float32')
if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()