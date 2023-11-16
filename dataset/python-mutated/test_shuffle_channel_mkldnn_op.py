import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
from paddle.base import core

@OpTestTool.skip_if_not_cpu_bf16()
class TestShuffleChannelOneDNNOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'shuffle_channel'
        self.set_dtype()
        self.set_group()
        self.inputs = {'X': np.random.random((5, 64, 2, 3)).astype(self.dtype)}
        self.attrs = {'use_mkldnn': True, 'group': self.group}
        (_, c, h, w) = self.inputs['X'].shape
        input_reshaped = np.reshape(self.inputs['X'], (-1, self.group, c // self.group, h, w))
        input_transposed = np.transpose(input_reshaped, (0, 2, 1, 3, 4))
        self.outputs = {'Out': np.reshape(input_transposed, (-1, c, h, w))}

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def set_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.group = 4

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace())

class TestShuffleChannelSingleGroupOneDNNOp(TestShuffleChannelOneDNNOp):

    def set_group(self):
        if False:
            i = 10
            return i + 15
        self.group = 1

class TestShuffleChannelBF16OneDNNOp(TestShuffleChannelOneDNNOp):

    def set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()