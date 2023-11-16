import unittest
import numpy as np
from mkldnn_op_test import format_reorder
from op_test import OpTest
from paddle.base import core

class TestTransposeOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_op_type()
        self.initTestCase()
        self.initInputData()
        self.use_mkldnn = True
        self._cpu_only = True
        self.axis = (0, 2, 3, 1)
        self.inputs = {'X': format_reorder(self.input_data, self.shape).astype(np.int8)}
        self.attrs = {'axis': list(self.axis), 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'XShape': np.random.random(self.shape).astype(np.int8), 'Out': self.inputs['X'].transpose(self.axis)}

    def init_op_type(self):
        if False:
            print('Hello World!')
        self.op_type = 'transpose2'

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CPUPlace(), 1e-05, no_check_set=['XShape'], check_dygraph=False)

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (2, 3, 4, 5)

    def initInputData(self):
        if False:
            print('Hello World!')
        self.input_data = (np.random.randint(0, 100, self.shape) - 50).astype(np.int8)

class TestINT8Case(TestTransposeOp):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (2, 4, 6, 8)

    def initInputData(self):
        if False:
            return 10
        self.input_data = (np.random.randint(0, 100, self.shape) - 50).astype(np.int8)

class TestUINT8Case(TestTransposeOp):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = (1, 3, 5, 7)

    def initDataType(self):
        if False:
            while True:
                i = 10
        self.input_data = np.random.randint(0, 100, self.shape).astype(np.uint8)
if __name__ == '__main__':
    unittest.main()