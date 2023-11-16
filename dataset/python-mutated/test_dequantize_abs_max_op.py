import math
import unittest
import numpy as np
from op_test import OpTest

def quantize_max_abs(x, max_range):
    if False:
        while True:
            i = 10
    scale = np.max(np.abs(x).flatten())
    y = np.round(x / scale * max_range)
    return (y, scale)

def dequantize_max_abs(x, scale, max_range):
    if False:
        print('Hello World!')
    y = scale / max_range * x
    return y

class TestDequantizeMaxAbsOp(OpTest):

    def set_args(self):
        if False:
            print('Hello World!')
        self.num_bits = 8
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = 'int8'

    def setUp(self):
        if False:
            return 10
        self.set_args()
        self.op_type = 'dequantize_abs_max'
        x = np.random.randn(31, 65).astype(self.data_type)
        (yq, scale) = quantize_max_abs(x, self.max_range)
        ydq = dequantize_max_abs(yq, scale, self.max_range)
        self.inputs = {'X': np.array(yq).astype(self.data_type), 'Scale': np.array(scale).astype('float32')}
        self.attrs = {'max_range': self.max_range}
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

class TestDequantizeMaxAbsOp5Bits(TestDequantizeMaxAbsOp):

    def set_args(self):
        if False:
            print('Hello World!')
        self.num_bits = 5
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = 'int8'

class TestDequantizeMaxAbsOpInt16(TestDequantizeMaxAbsOp):

    def set_args(self):
        if False:
            return 10
        self.num_bits = 16
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = 'int16'
if __name__ == '__main__':
    unittest.main()