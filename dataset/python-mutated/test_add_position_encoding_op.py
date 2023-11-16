import math
import unittest
import numpy as np
from op_test import OpTest

def add_position_encoding(input, alpha=1.0, beta=1.0):
    if False:
        for i in range(10):
            print('nop')
    batch_size = input.shape[0]
    max_length = input.shape[1]
    enc_size = input.shape[2]
    out = np.copy(input)
    half_shape = int(enc_size / 2)
    for i in range(batch_size):
        for j in range(max_length):
            for k in range(half_shape):
                val = j / pow(10000.0, k * 1.0 / (half_shape - 1)) if half_shape > 1 else j / 10000.0
                out[i, j, k] = input[i, j, k] * alpha + math.sin(val) * beta
                out[i, j, half_shape + k] = input[i, j, half_shape + k] * alpha + math.cos(val) * beta
    return out

class TestAddPositionEncodingTensorOp(OpTest):
    """
    This class is to test the AddPositionEncodingOp
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        the prepared section for add position encoding op\n        '
        self.op_type = 'add_position_encoding'
        self.dtype = np.float64
        self.init_input_output()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {'Out': self.out}
        self.attrs = {'alpha': self.alpha, 'beta': self.beta}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        check the correctness of output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        check the correctness of grad\n        '
        self.check_grad(['X'], 'Out', check_dygraph=False)

    def init_input_output(self):
        if False:
            return 10
        '\n        init the input and output for test cases\n        '
        self.alpha = 0.6
        self.beta = 0.5
        self.x = np.random.uniform(0.1, 1, [2, 15, 4]).astype(self.dtype)
        self.out = add_position_encoding(self.x, self.alpha, self.beta)

class TestAddPositionEncodingLoDTensorOp(OpTest):
    """
    This class is to test the AddPositionEncodingLoDTensorOp
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        the prepared section for add position encoding LoDTensor op\n        '
        self.op_type = 'add_position_encoding'
        self.dtype = np.float64
        self.init_input_output()
        self.inputs = {'X': (self.x, self.lod)}
        self.outputs = {'Out': (self.out, self.lod)}
        self.attrs = {'alpha': self.alpha, 'beta': self.beta}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        check the correctness of output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        '\n        check the correctness of grad\n        '
        self.check_grad(['X'], 'Out', check_dygraph=False)

    def init_input_output(self):
        if False:
            i = 10
            return i + 15
        '\n        init the input and output for test cases\n        '
        self.alpha = 0.6
        self.beta = 0.5
        self.x = np.random.uniform(0.1, 1, [20, 6]).astype(self.dtype)
        self.lod = [[13, 7]]
        self.out = np.copy(self.x)
        batch_size = len(self.lod[0])
        enc_size = self.x.shape[1]
        start = 0
        half_shape = int(enc_size / 2)
        for i in range(batch_size):
            max_length = self.lod[0][i]
            for j in range(max_length):
                for k in range(half_shape):
                    val = j / pow(10000.0, k * 1.0 / (half_shape - 1)) if half_shape > 1 else j / 10000.0
                    pos = start + j
                    self.out[pos, k] = self.x[pos, k] * self.alpha + math.sin(val) * self.beta
                    self.out[pos, half_shape + k] = self.x[pos, half_shape + k] * self.alpha + math.cos(val) * self.beta
            start += max_length
if __name__ == '__main__':
    unittest.main()