import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestReshapeBf16Op(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reshape2'
        self.use_mkldnn = False
        self.mkldnn_data_type = 'bfloat16'
        self.init_data()
        self.init_input_data()
        self.inputs = {'X': self.input_data}
        self.attrs = {'shape': self.new_shape, 'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype(np.float32)}

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.input_data_fp32 = np.random.random(self.ori_shape).astype(np.float32)
        self.input_data = convert_float_to_uint16(self.input_data_fp32)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'], check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', check_dygraph=False, user_defined_grads=[self.input_data_fp32], user_defined_grad_outputs=[self.inputs['X'].reshape(self.infered_shape)])
if __name__ == '__main__':
    enable_static()
    unittest.main()