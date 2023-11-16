import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from test_conv2d_transpose_op import conv2dtranspose_forward_naive
from paddle import enable_static
from paddle.base import core

def conv2d_bias_naive(out, bias):
    if False:
        for i in range(10):
            print('nop')
    (_, out_c, _, _) = out.shape
    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestConv2DTransposeBF16MKLDNNOp(OpTest):

    def test_check_output(self):
        if False:
            return 10
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            return 10
        pass

    def test_check_grad_no_input(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_no_filter(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def init_op_type(self):
        if False:
            print('Hello World!')
        self.data_format = 'NCHW'
        self.op_type = 'conv2d_transpose'
        self._cpu_only = True

    def init_test_case(self):
        if False:
            return 10
        self.pad = [0, 0]
        self.fuse_bias = False
        self.use_mkldnn = True
        self.is_test = True
        self.bias_size = None
        self.fuse_activation = ''
        self.fuse_alpha = 0.0
        self.fuse_beta = 0.0
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3]
        self.groups = 1
        self.output_size = None
        self.output_padding = []
        self.data_format = 'NCHW'
        self.pad = [0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.force_fp32_output = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_type = np.uint16
        self.dtype = np.uint16
        self.mkldnn_data_type = 'bfloat16'
        self.init_op_type()
        self.init_test_case()
        input = np.random.random(self.input_size).astype(np.float32)
        filter = np.random.random(self.filter_size).astype(np.float32)
        self.attrs = {'strides': self.stride, 'paddings': self.pad, 'padding_algorithm': self.padding_algorithm, 'groups': self.groups, 'dilations': self.dilations, 'is_test': self.is_test, 'use_mkldnn': self.use_mkldnn, 'mkldnn_data_type': self.mkldnn_data_type, 'force_fp32_output': self.force_fp32_output, 'data_format': self.data_format, 'fuse_activation': self.fuse_activation, 'fuse_alpha': self.fuse_alpha, 'fuse_beta': self.fuse_beta}
        if self.output_size is not None:
            self.attrs['output_size'] = self.output_size
        if len(self.output_padding) > 0:
            self.attrs['output_padding'] = self.output_padding
        output = conv2dtranspose_forward_naive(input, filter, self.attrs).astype(np.float32)
        if self.input_type is not np.float32:
            input = convert_float_to_uint16(input)
        self.inputs = {'Input': input.view(self.input_type), 'Filter': OpTest.np_dtype_to_base_dtype(filter)}
        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(np.float32)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(np.float32)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_base_dtype(bias)
        if self.fuse_activation == 'relu':
            output = np.maximum(output, 0).astype(np.float32)
        output = output.astype(np.float32)
        if not self.force_fp32_output:
            output = convert_float_to_uint16(output, self.attrs['data_format'])
        self.outputs['Output'] = output

class TestMKLDNNFuseBias(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        super().init_test_case()
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]

class TestMKLDNNWithPad(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        super().init_test_case()
        self.pad = [1, 1]
        self.input_size = [2, 3, 10, 10]

class TestMKLDNNWithStride(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        super().init_test_case()
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]

class TestMKLDNNWithAsymPad(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        super().init_test_case()
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestMKLDNNWithSamePad(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        super().init_test_case()
        self.pad = [0, 0]
        self.padding_algorithm = 'SAME'

class TestMKLDNNWithValidPad(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        super().init_test_case()
        self.pad = [1, 1]
        self.padding_algorithm = 'VALID'

class TestMKLDNNWithValidPad_NHWC(TestMKLDNNWithValidPad):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        super().init_test_case()
        self.data_format = 'NHWC'
        (N, C, H, W) = self.input_size
        self.input_size = [N, H, W, C]

class TestConv2DTransposeMKLDNNWithDilationsExplicitPad(TestConv2DTransposeBF16MKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        super().init_test_case()
        self.stride = [2, 1]
        self.dilations = [1, 2]
        self.groups = 1
        self.input_size = [4, 3, 8, 7]
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 4, 3]
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = 'EXPLICIT'
if __name__ == '__main__':
    enable_static()
    unittest.main()