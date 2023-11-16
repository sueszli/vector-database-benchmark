import unittest
import numpy as np
from op_test import OpTest
from test_conv2d_transpose_op import TestConv2DTransposeOp
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

class TestConv2DTransposeMKLDNNOp(TestConv2DTransposeOp):

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        return

    def test_check_grad_no_input(self):
        if False:
            return 10
        return

    def test_check_grad_no_filter(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def test_check_output(self):
        if False:
            return 10
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-05, check_dygraph=not self.use_mkldnn)
        else:
            self.check_output(check_dygraph=not self.use_mkldnn)

    def init_op_type(self):
        if False:
            while True:
                i = 10
        self.data_format = 'NCHW'
        self.op_type = 'conv2d_transpose'
        self._cpu_only = True

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_mkldnn = True
        self.is_test = True
        self.pad = [0, 0]
        self.fuse_bias = False
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
        self.dtype = np.float32

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        TestConv2DTransposeOp.setUp(self)
        output = self.outputs['Output']
        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(self.dtype)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(self.dtype)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_base_dtype(bias)
        if self.fuse_activation == 'relu':
            output = np.maximum(output, 0).astype(self.dtype)
        output = output.astype(self.dtype)
        self.attrs['fuse_activation'] = self.fuse_activation
        self.attrs['fuse_alpha'] = self.fuse_alpha
        self.attrs['fuse_beta'] = self.fuse_beta
        self.attrs['mkldnn_data_type'] = 'float32'
        self.attrs['force_fp32_output'] = False
        self.outputs['Output'] = output

class TestMKLDNNFuseBias(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]

class TestMKLDNNWithPad(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.input_size = [2, 3, 10, 10]

class TestMKLDNNWithStride(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]

class TestMKLDNNWithAsymPad(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestMKLDNNWithSamePad(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [0, 0]
        self.padding_algorithm = 'SAME'

class TestMKLDNNWithValidPad(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.padding_algorithm = 'VALID'

class TestMKLDNNWithValidPad_NHWC(TestMKLDNNWithValidPad):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        super().init_test_case()
        self.data_format = 'NHWC'
        (N, C, H, W) = self.input_size
        self.input_size = [N, H, W, C]

class TestConv2DTransposeMKLDNNWithDilationsExplicitPad(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.stride = [2, 1]
        self.dilations = [1, 2]
        self.groups = 1
        self.input_size = [4, 3, 8, 7]
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 4, 3]
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = 'EXPLICIT'

class TestMKLDNNWithGroups(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.groups = 2
        self.input_size = [2, 4, 5, 5]
        f_c = self.input_size[1]
        self.filter_size = [f_c, 3, 3, 3]

class TestMKLDNNWithGroups_NHWC(TestConv2DTransposeMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.groups = 2
        self.input_size = [2, 5, 5, 4]
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 3, 3, 3]
        self.data_format = 'NHWC'
if __name__ == '__main__':
    enable_static()
    unittest.main()