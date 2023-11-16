import unittest
import numpy as np
from op_test import convert_float_to_uint16
from test_sum_op import TestSumOp
from paddle import enable_static
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestSumBF16MKLDNN(TestSumOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'sum'
        self.use_mkldnn = True
        self.mkldnn_data_type = 'bfloat16'
        x0 = np.random.random((25, 8)).astype('float32')
        x1 = np.random.random((25, 8)).astype('float32')
        x2 = np.random.random((25, 8)).astype('float32')
        x0_bf16 = convert_float_to_uint16(x0)
        x1_bf16 = convert_float_to_uint16(x1)
        x2_bf16 = convert_float_to_uint16(x2)
        self.inputs = {'X': [('x0', x0_bf16), ('x1', x1_bf16), ('x2', x2_bf16)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': convert_float_to_uint16(y)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()