import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
from test_log_softmax import ref_log_softmax
import paddle
from paddle.base import core

class TestLogSoftmaxOneDNNOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'log_softmax'
        self.set_dtype()
        self.set_shape()
        self.set_axis()
        x = np.random.uniform(0.1, 1.0, self.shape).astype(np.float32)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x) if len(self.shape) > 0 else np.array(0.0).astype(self.dtype)
        if self.dtype == np.uint16:
            x = convert_float_to_uint16(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}

    def set_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32

    def set_shape(self):
        if False:
            return 10
        self.shape = [2, 3, 4, 5]

    def set_axis(self):
        if False:
            i = 10
            return i + 15
        self.axis = -1

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

class TestLogSoftmax0DOneDNNOp(TestLogSoftmaxOneDNNOp):

    def set_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = []

class TestLogSoftmax1DOneDNNOp(TestLogSoftmaxOneDNNOp):

    def set_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [100]

class TestLogSoftmax3DOneDNNOp(TestLogSoftmaxOneDNNOp):

    def set_shape(self):
        if False:
            print('Hello World!')
        self.shape = [12, 10, 3]

class TestLogSoftmax5DOneDNNOp(TestLogSoftmaxOneDNNOp):

    def set_shape(self):
        if False:
            print('Hello World!')
        self.shape = [2, 3, 4, 5, 6]

class TestLogSoftmaxPositiveAxisOneDNNOp(TestLogSoftmaxOneDNNOp):

    def set_axis(self):
        if False:
            return 10
        self.axis = 2

@OpTestTool.skip_if_not_cpu_bf16()
class TestLogSoftmax1DBF16OneDNNOp(TestLogSoftmax1DOneDNNOp):

    def set_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16

@OpTestTool.skip_if_not_cpu_bf16()
class TestLogSoftmaxPositiveAxisBF16OneDNNOp(TestLogSoftmaxPositiveAxisOneDNNOp):

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

@OpTestTool.skip_if_not_cpu_bf16()
class TestLogSoftmax5DBF16OneDNNOp(TestLogSoftmax5DOneDNNOp):

    def set_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.uint16
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()