import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestCastBF16ToFP32MKLDNNOp(OpTest):

    def init_data(self):
        if False:
            return 10
        self.out = np.random.random(size=self.shape).astype('float32')
        self.x = convert_float_to_uint16(self.out)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_shape()
        self.init_data()
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}
        prepare_dtype = lambda x: int(core.VarDesc.VarType.BF16 if x.dtype != np.float32 else core.VarDesc.VarType.FP32)
        self.attrs = {'in_dtype': prepare_dtype(self.x), 'out_dtype': prepare_dtype(self.out), 'use_mkldnn': True}
        self.op_type = 'cast'

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', check_dygraph=False, user_defined_grads=[self.inputs['X']], user_defined_grad_outputs=[self.outputs['Out']])

    def init_shape(self):
        if False:
            print('Hello World!')
        self.shape = [10, 10]

class TestCastFP32ToBF16MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        if False:
            print('Hello World!')
        self.x = np.random.random(size=[2, 6]).astype('float32')
        self.out = convert_float_to_uint16(self.x)

class TestCastBF16ToBF16MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random(size=[6, 13]).astype('uint16')
        self.out = self.x

class TestCastFP32ToFP32MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random(size=[7, 15]).astype('float32')
        self.out = self.x

class TestCastBF16ToFP32MKLDNNOp_ZeroDim(TestCastBF16ToFP32MKLDNNOp):

    def init_shape(self):
        if False:
            print('Hello World!')
        self.shape = []
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()