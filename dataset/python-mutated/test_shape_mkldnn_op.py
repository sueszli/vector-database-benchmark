import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
from paddle.base import core

class TestShape3DFP32OneDNNOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'shape'
        self.python_api = paddle.tensor.shape
        self.config()
        self.attrs = {'use_mkldnn': True}
        self.inputs = {'Input': np.zeros(self.shape).astype(self.dtype)}
        self.outputs = {'Out': np.array(self.shape)}

    def config(self):
        if False:
            print('Hello World!')
        self.shape = [5, 7, 4]
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            return 10
        self.check_output_with_place(core.CPUPlace())

class TestShape0DFP32OneDNNOp(TestShape3DFP32OneDNNOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.shape = []
        self.dtype = np.float32

@OpTestTool.skip_if_not_cpu_bf16()
class TestShape6DBF16OneDNNOp(TestShape3DFP32OneDNNOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.shape = [10, 2, 3, 4, 5, 2]
        self.dtype = np.uint16

class TestShape9DINT8OneDNNOp(TestShape3DFP32OneDNNOp):

    def config(self):
        if False:
            while True:
                i = 10
        self.shape = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.dtype = np.int8

class TestShape2DUINT8OneDNNOp(TestShape3DFP32OneDNNOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.shape = [7, 11]
        self.dtype = np.uint8
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()