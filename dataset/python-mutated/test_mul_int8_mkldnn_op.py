import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle.base import core
'\n test case for s8 * s8\n'

@skip_check_grad_ci(reason='mul_mkldnn_op does not implement grad operator, check_grad is not required.')
class TestMKLDNNMulOpS8S8(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'mul'
        self.init_kernel_type()
        self.init_data_type()
        self.init_data()
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'scale_x': self.scale_x, 'scale_y': self.scale_y, 'scale_out': self.scale_out, 'force_fp32_output': self.force_fp32}

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True
        self.force_fp32 = True

    def init_data_type(self):
        if False:
            print('Hello World!')
        self.srctype = np.uint8
        self.dsttype = np.float32 if self.force_fp32 else np.int8

    def init_data(self):
        if False:
            print('Hello World!')
        self.scale_x = 0.6
        self.scale_y = [0.8]
        self.scale_out = 1.0
        if self.srctype == np.int8:
            A_data = np.random.randint(-127, 127, (20, 5)).astype(np.int8)
        else:
            A_data = np.random.randint(0, 127, (20, 5)).astype(np.uint8)
        B_data = np.random.uniform(-127, 127, (5, 20)).astype(np.float32)
        quant_B = np.round(B_data * self.scale_y[0]).astype(np.int_)
        output = np.dot(A_data, quant_B)
        scale_output_shift = self.scale_out / (self.scale_x * self.scale_y[0])
        if self.force_fp32:
            output = (output * scale_output_shift).astype(self.dsttype)
        else:
            output = np.round(output * scale_output_shift).astype(self.dsttype)
        self.inputs = {'X': A_data, 'Y': B_data}
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output_with_place(core.CPUPlace(), atol=0, check_dygraph=False)
'\n test case for  s8 * u8\n'

class TestMKLDNNMulOpS8U8(TestMKLDNNMulOpS8S8):

    def init_data_type(self):
        if False:
            while True:
                i = 10
        self.srctype = np.uint8
        self.dsttype = np.float32 if self.force_fp32 else np.int8
'\n test case for  s8 * s8\n'

class TestMKLDNNMulOpS8S8WithFlatten(TestMKLDNNMulOpS8S8):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'mul'
        self.init_kernel_type()
        self.init_data_type()
        self.init_data()
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'scale_x': self.scale_x, 'scale_y': self.scale_y, 'scale_out': self.scale_out, 'force_fp32_output': self.force_fp32, 'x_num_col_dims': 2, 'y_num_col_dims': 2}

    def init_data(self):
        if False:
            print('Hello World!')
        self.scale_x = 0.6
        self.scale_y = [0.8]
        self.scale_out = 1.0
        if self.srctype == np.int8:
            A_data = np.random.randint(-127, 127, (3, 4, 4, 3)).astype(np.int8)
        else:
            A_data = np.random.randint(0, 127, (3, 4, 4, 3)).astype(np.uint8)
        B_data = np.random.uniform(-127, 127, (2, 6, 1, 2, 3)).astype(np.float32)
        A_data_reshape = A_data.reshape(3 * 4, 4 * 3)
        B_data_reshape = B_data.reshape(2 * 6, 1 * 2 * 3)
        quant_B = np.round(B_data_reshape * self.scale_y[0]).astype(np.int_)
        output = np.dot(A_data_reshape, quant_B)
        scale_output_shift = self.scale_out / (self.scale_x * self.scale_y[0])
        if self.force_fp32:
            output = (output * scale_output_shift).astype(self.dsttype)
        else:
            output = np.round(output * scale_output_shift).astype(self.dsttype)
        output = output.reshape(3, 4, 1, 2, 3)
        self.inputs = {'X': A_data, 'Y': B_data}
        self.outputs = {'Out': output}
'\n test case for  s8 * u8\n'

class TestMKLDNNMulOpS8U8WithFlatten(TestMKLDNNMulOpS8S8WithFlatten):

    def init_data_type(self):
        if False:
            i = 10
            return i + 15
        self.srctype = np.uint8
        self.dsttype = np.float32 if self.force_fp32 else np.int8
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()