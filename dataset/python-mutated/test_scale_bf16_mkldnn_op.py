import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
@unittest.skipIf(core.is_compiled_with_cuda(), 'core is compiled with CUDA which has no BF implementation')
class TestScaleOpBF16(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'scale'
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale = -2.3
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'scale': self.scale, 'use_mkldnn': True, 'bias': 0.4}
        self.use_mkldnn = True
        self.outputs = {'Out': self.x_fp32 * self.attrs['scale'] + self.attrs['bias']}

    def calculate_grads(self):
        if False:
            for i in range(10):
                print('nop')
        bias = 0
        if 'bias' in self.attrs:
            bias = self.attrs['bias']
        scale = self.scale
        if 'ScaleTensor' in self.attrs:
            scale = self.attrs['ScaleTensor']
        self.out = self.x_fp32 * scale + bias
        self.dx = self.out * scale

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.calculate_grads()
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', check_dygraph=False, user_defined_grads=[self.dx], user_defined_grad_outputs=[convert_float_to_uint16(self.out)])

class TestScaleOpBF16BiasNotAfterScale(TestScaleOpBF16):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'scale'
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale = 1.5
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'scale': self.scale, 'use_mkldnn': True, 'bias': 0.0, 'bias_after_scale': False}
        self.use_mkldnn = True
        self.outputs = {'Out': (self.x_fp32 + self.attrs['bias']) * self.attrs['scale']}

class TestScaleOpBF16ScaleTensor(TestScaleOpBF16):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'scale'
        self.scale = -2.3
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale_tensor = np.array([self.scale]).astype(np.float32)
        self.inputs = {'X': self.x_bf16, 'ScaleTensor': convert_float_to_uint16(self.scale_tensor)}
        self.attrs = {'use_mkldnn': True}
        self.outputs = {'Out': self.x_fp32 * self.scale}

class TestScaleOpBF16ScaleTensorNotBiasAfterScale(TestScaleOpBF16):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'scale'
        self.scale = 1.2
        self.x_fp32 = np.random.random((9, 13)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale_tensor = np.array([self.scale]).astype(np.float32)
        self.inputs = {'X': self.x_bf16, 'ScaleTensor': convert_float_to_uint16(self.scale_tensor)}
        self.attrs = {'bias': -1.1, 'bias_after_scale': False, 'use_mkldnn': True}
        self.outputs = {'Out': (self.x_fp32 + self.attrs['bias']) * self.scale}
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()