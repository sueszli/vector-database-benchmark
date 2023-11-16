import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestTransposeOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'transpose2'
        self.use_mkldnn = True
        self.mkldnn_data_type = 'bfloat16'
        self.init_test_case()
        self.init_test_data()
        self.axis = (0, 2, 3, 1)
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': list(self.axis), 'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type}
        self.outputs = {'XShape': np.random.random(self.shape).astype(np.uint16), 'Out': self.inputs['X'].transpose(self.axis)}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'])

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shape = (2, 3, 4, 5)

    def init_test_data(self):
        if False:
            while True:
                i = 10
        self.input_data = convert_float_to_uint16(np.random.random(self.shape).astype(np.float32))

class TestBF16Case(TestTransposeOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shape = (2, 4, 6, 8)
if __name__ == '__main__':
    enable_static()
    unittest.main()