import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
from paddle.base import core

@OpTestTool.skip_if_not_cpu()
class TestStack2DOneDNNOp(OpTest):

    def initDefaultParameters(self):
        if False:
            return 10
        self.num_inputs = 4
        self.input_dim = (2, 2)
        self.axis = 1
        self.dtype = np.float32

    def initParameters(self):
        if False:
            return 10
        pass

    def getInputNames(self):
        if False:
            return 10
        input_names = []
        for i in range(self.num_inputs):
            input_names.append(f'x{i}')
        return input_names

    def setUp(self):
        if False:
            return 10
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.op_inputs = []
        for i in range(self.num_inputs):
            self.op_inputs.append(np.random.random(size=self.input_dim).astype(np.float32))
        input_list = []
        input_names = self.getInputNames()
        for i in range(self.num_inputs):
            input_list.append((input_names[i], self.op_inputs[i]))
        self.inputs = {'X': input_list}
        self.outputs = {'Y': np.stack(self.op_inputs, axis=self.axis)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestStack1DOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            return 10
        self.input_dim = 100
        self.axis = 0

class TestStack0DOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.input_dim = ()
        self.axis = 0

class TestStack1DAxis1OneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.input_dim = 100
        self.axis = 1

class TestStack2DAxisLastOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            return 10
        self.input_dim = (13, 24)
        self.num_inputs = 5
        self.axis = -1

class TestStack3DAxisNegativeOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_dim = (10, 128, 128)
        self.axis = -2

class TestStack3DOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.input_dim = (10, 128, 128)
        self.num_inputs = 3
        self.axis = 1

class TestStack4DOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            return 10
        self.input_dim = (2, 2, 2, 2)
        self.num_inputs = 3
        self.axis = 4

class TestStack5DOneDNNOp(TestStack2DOneDNNOp):

    def initParameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_dim = (2, 3, 4, 5, 6)
        self.num_inputs = 6
        self.axis = 0
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()