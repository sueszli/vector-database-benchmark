import unittest
import numpy as np
from op_test import OpTest
from paddle import enable_static
from paddle.base import core

class TestConcatAxis0OneDNNOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'concat'
        self.mkldnn_data_type = 'float32'
        self.init_axis()
        self.init_shape()
        self.init_test_data()
        self.configure_datatype()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True, 'mkldnn_data_type': self.mkldnn_data_type}
        self.output = np.concatenate((self.x0, self.x1, self.x2), axis=self.axis).astype(self.dtype)
        self.outputs = {'Out': self.output}

    def configure_datatype(self):
        if False:
            while True:
                i = 10
        self.mkldnn_data_type = 'float32'
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['x0'], 'Out', check_dygraph=False)
        self.check_grad(['x1'], 'Out', check_dygraph=False)
        self.check_grad(['x2'], 'Out', check_dygraph=False)

    def init_test_data(self):
        if False:
            print('Hello World!')
        self.x0 = np.random.random(self.x0_shape).astype(np.float32)
        self.x1 = np.random.random(self.x1_shape).astype(np.float32)
        self.x2 = np.random.random(self.x2_shape).astype(np.float32)

    def init_axis(self):
        if False:
            for i in range(10):
                print('nop')
        self.axis = 0

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x0_shape = [2, 2, 1, 50]
        self.x1_shape = [1, 2, 1, 50]
        self.x2_shape = [3, 2, 1, 50]

class TestConcatAxis1OneDNNOp(TestConcatAxis0OneDNNOp):

    def init_axis(self):
        if False:
            while True:
                i = 10
        self.axis = 1

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x0_shape = [1, 1, 5, 50]
        self.x1_shape = [1, 2, 5, 50]
        self.x2_shape = [1, 3, 5, 50]

class TestConcatAxis2OneDNNOp(TestConcatAxis0OneDNNOp):

    def init_axis(self):
        if False:
            i = 10
            return i + 15
        self.axis = 2

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x0_shape = [2, 3, 4, 50]
        self.x1_shape = [2, 3, 5, 50]
        self.x2_shape = [2, 3, 6, 50]

class TestConcatAxis3OneDNNOp(TestConcatAxis0OneDNNOp):

    def init_axis(self):
        if False:
            print('Hello World!')
        self.axis = 3

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x0_shape = [5, 3, 5, 5]
        self.x1_shape = [5, 3, 5, 6]
        self.x2_shape = [5, 3, 5, 7]
if __name__ == '__main__':
    enable_static()
    unittest.main()