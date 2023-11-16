import unittest
import numpy as np
from op_test import convert_float_to_uint16
from test_softmax_op import TestSoftmaxOp, TestSoftmaxOp2, TestSoftmaxOp3, TestSoftmaxOp4, TestSoftmaxOp5, TestSoftmaxOp6
from paddle import enable_static
from paddle.base import core

def stable_softmax(x):
    if False:
        while True:
            i = 10
    'Compute the softmax of vector x in a numerically stable way.'
    shiftx = x - np.max(x).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestSoftmaxMKLDNNOp(TestSoftmaxOp):

    def get_x_shape(self):
        if False:
            while True:
                i = 10
        return [10, 10]

    def get_axis(self):
        if False:
            for i in range(10):
                print('nop')
        return -1

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'softmax'
        self.use_mkldnn = True
        self.dtype = np.uint16
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()
        x = np.random.uniform(0.1, 1, self.shape).astype(np.float64)
        out = convert_float_to_uint16(np.apply_along_axis(stable_softmax, self.axis, x))
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            return 10
        pass

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp2(TestSoftmaxOp2):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp3(TestSoftmaxOp3):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp4(TestSoftmaxOp4):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp5(TestSoftmaxOp5):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp6(TestSoftmaxOp6):

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True
if __name__ == '__main__':
    enable_static()
    unittest.main()