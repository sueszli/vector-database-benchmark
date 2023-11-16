import unittest
import numpy as np
from op_test import OpTest

class TestShuffleChannelOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'shuffle_channel'
        self.batch_size = 10
        self.input_channels = 16
        self.layer_h = 4
        self.layer_w = 4
        self.group = 4
        self.x = np.random.random((self.batch_size, self.input_channels, self.layer_h, self.layer_w)).astype('float32')
        self.inputs = {'X': self.x}
        self.attrs = {'group': self.group}
        (n, c, h, w) = self.x.shape
        input_reshaped = np.reshape(self.x, (-1, self.group, c // self.group, h, w))
        input_transposed = np.transpose(input_reshaped, (0, 2, 1, 3, 4))
        self.outputs = {'Out': np.reshape(input_transposed, (-1, c, h, w))}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')
if __name__ == '__main__':
    unittest.main()