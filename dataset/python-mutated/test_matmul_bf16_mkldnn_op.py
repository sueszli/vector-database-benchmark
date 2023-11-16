import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestMatmulBf16MklDNNOp(OpTest):

    def generate_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_fp32 = np.random.random((25, 2, 2)).astype(np.float32)
        self.y_fp32 = np.random.random((25, 2, 2)).astype(np.float32)
        self.out = self.alpha * np.matmul(self.x_fp32, self.y_fp32)

    def set_attributes(self):
        if False:
            return 10
        self.attrs = {'alpha': self.alpha, 'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type, 'force_fp32_output': self.force_fp32_output, 'transpose_X': False, 'transpose_Y': False}

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'matmul'
        self.alpha = 1.0
        self.use_mkldnn = True
        self.dtype = np.uint16
        self.mkldnn_data_type = 'bfloat16'
        self.force_fp32_output = False
        self.generate_data()
        self.set_attributes()
        if not self.force_fp32_output:
            self.out = convert_float_to_uint16(self.out)
        self.outputs = {'Out': self.out}
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.y_bf16 = convert_float_to_uint16(self.y_fp32)
        self.inputs = {'X': self.x_bf16, 'Y': self.y_bf16}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.calculate_grads()
        self.check_grad_with_place(core.CPUPlace(), ['X', 'Y'], 'Out', check_dygraph=False, user_defined_grads=[self.dx, self.dy], user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    def matmul_grad(self, x, transpose_x, y, transpose_y):
        if False:
            i = 10
            return i + 15
        x_transpose_axes = [1, 0] if x.ndim == 2 else [0, 2, 1]
        y_transpose_axes = [1, 0] if y.ndim == 2 else [0, 2, 1]
        x = np.transpose(x, x_transpose_axes) if transpose_x else x
        y = np.transpose(y, y_transpose_axes) if transpose_y else y
        return self.alpha * np.matmul(x, y)

    def calculate_grads(self):
        if False:
            print('Hello World!')
        x_transpose_axes = [1, 0] if self.x_fp32.ndim == 2 else [0, 2, 1]
        y_transpose_axes = [1, 0] if self.y_fp32.ndim == 2 else [0, 2, 1]
        x = np.transpose(self.x_fp32, x_transpose_axes) if self.attrs['transpose_X'] is True else self.x_fp32
        y = np.transpose(self.y_fp32, y_transpose_axes) if self.attrs['transpose_Y'] is True else self.y_fp32
        dout = self.alpha * np.matmul(x, y)
        if self.attrs['transpose_X'] is True and self.attrs['transpose_Y'] is True:
            self.dx = self.matmul_grad(self.y_fp32, True, dout, True)
            self.dy = self.matmul_grad(dout, True, self.x_fp32, True)
        elif self.attrs['transpose_X'] is True and self.attrs['transpose_Y'] is False:
            self.dx = self.matmul_grad(self.y_fp32, False, dout, True)
            self.dy = self.matmul_grad(self.x_fp32, False, dout, False)
        elif self.attrs['transpose_X'] is False and self.attrs['transpose_Y'] is True:
            self.dx = self.matmul_grad(dout, False, self.y_fp32, False)
            self.dy = self.matmul_grad(dout, True, self.x_fp32, False)
        else:
            self.dx = self.matmul_grad(dout, False, self.y_fp32, True)
            self.dy = self.matmul_grad(self.x_fp32, True, dout, False)
        self.dout = dout

class TestDnnlMatMulOpAlpha(TestMatmulBf16MklDNNOp):

    def generate_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_fp32 = np.random.random((17, 2, 3)).astype(np.float32)
        self.y_fp32 = np.random.random((17, 3, 2)).astype(np.float32)
        self.alpha = 2.0
        self.out = self.alpha * np.matmul(self.x_fp32, self.y_fp32)

class TestDnnlMatMulOp2D(TestMatmulBf16MklDNNOp):

    def generate_data(self):
        if False:
            return 10
        self.x_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.y_fp32 = np.random.random((9, 12)).astype(np.float32)
        self.out = np.matmul(self.x_fp32, self.y_fp32)

class TestDnnlMatMulOpTransposeX(TestMatmulBf16MklDNNOp):

    def generate_data(self):
        if False:
            i = 10
            return i + 15
        self.x_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.y_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.out = np.matmul(np.transpose(self.x_fp32), self.y_fp32)

    def set_attributes(self):
        if False:
            while True:
                i = 10
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type, 'transpose_X': True, 'transpose_Y': False}

class TestDnnlMatMulOpTransposeY(TestMatmulBf16MklDNNOp):

    def generate_data(self):
        if False:
            i = 10
            return i + 15
        self.x_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.y_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.out = np.matmul(self.x_fp32, np.transpose(self.y_fp32))

    def set_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type, 'transpose_Y': True, 'transpose_X': False}

class TestMatmulBf16MklDNNForceFp32Output(TestMatmulBf16MklDNNOp):

    def generate_data(self):
        if False:
            while True:
                i = 10
        self.x_fp32 = np.random.random((12, 9)).astype(np.float32)
        self.y_fp32 = np.random.random((9, 12)).astype(np.float32)
        self.force_fp32_output = True
        self.alpha = 0.5
        self.out = self.alpha * np.matmul(self.x_fp32, self.y_fp32)
if __name__ == '__main__':
    enable_static()
    unittest.main()