import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
from test_conv2d_op import TestConv2DOp, TestConv2DOp_v2

def conv2d_bias_naive(out, bias):
    if False:
        for i in range(10):
            print('nop')
    (_, out_c, _, _) = out.shape
    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out

def conv2d_residual_naive(out, residual):
    if False:
        while True:
            i = 10
    assert out.shape == residual.shape
    out = np.add(out, residual)
    return out

class TestConv2DMKLDNNOp(TestConv2DOp):

    def init_group(self):
        if False:
            return 10
        self.groups = 1

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.data_format = 'NCHW'
        self.use_mkldnn = True
        self._cpu_only = True
        self.dtype = np.float32

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fuse_bias = False
        self.bias_size = None
        self.fuse_activation = ''
        self.fuse_alpha = 0
        self.fuse_beta = 0
        self.fuse_residual_connection = False
        self.input_residual_size = None
        TestConv2DOp.setUp(self)
        output = self.outputs['Output']
        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(self.dtype)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(self.dtype)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_base_dtype(bias)
        if self.fuse_residual_connection and self.input_residual_size is not None:
            input_residual = np.random.random(self.input_residual_size).astype(self.dtype)
            output = conv2d_residual_naive(output, input_residual)
            self.attrs['fuse_residual_connection'] = self.fuse_residual_connection
            self.inputs['ResidualData'] = OpTest.np_dtype_to_base_dtype(input_residual)
        if self.fuse_activation == 'relu':
            output = np.maximum(output, 0).astype(self.dsttype)
        if self.fuse_activation == 'relu6':
            output = np.minimum(np.maximum(output, 0), self.fuse_beta).astype(self.dsttype)
        if self.fuse_activation != '' or self.fuse_bias or self.fuse_residual_connection:
            self.op_type = 'fused_conv2d'
        output = output.astype(self.dtype)
        self.attrs['fuse_bias'] = self.fuse_bias
        self.attrs['fuse_activation'] = self.fuse_activation
        self.attrs['fuse_alpha'] = self.fuse_alpha
        self.attrs['fuse_beta'] = self.fuse_beta
        self.attrs['fuse_residual_connection'] = self.fuse_residual_connection
        self.outputs['Output'] = output

@skip_check_grad_ci(reason='Fusion is for inference only, check_grad is not required.')
class TestWithbreluFusion(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        TestConv2DMKLDNNOp.init_test_case(self)
        self.fuse_activation = 'relu6'
        self.fuse_beta = 6.0
        self.dsttype = np.float32

@skip_check_grad_ci(reason='Fusion is for inference only, check_grad is not required.')
class TestWithFuse(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]
        self.fuse_residual_connection = True
        self.input_residual_size = [2, 6, 5, 5]

class TestWithPadWithBias(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.input_size = [2, 3, 6, 6]

class TestWithStride(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]

class TestWithGroup(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 6, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.groups = 3

class TestWith1x1(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        TestConv2DMKLDNNOp.init_test_case(self)
        self.filter_size = [40, 3, 1, 1]

class TestWithInput1x1Filter1x1(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            return 10
        TestConv2DMKLDNNOp.init_test_case(self)
        self.input_size = [2, 60, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3

class TestConv2DOp_AsyPadding_MKLDNN(TestConv2DOp_v2):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True
        self.dtype = np.float32

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestConv2DOp_Same_MKLDNN(TestConv2DOp_AsyPadding_MKLDNN):

    def init_paddings(self):
        if False:
            return 10
        self.pad = [0, 0]
        self.padding_algorithm = 'SAME'

class TestConv2DOp_Valid_MKLDNN(TestConv2DOp_AsyPadding_MKLDNN):

    def init_paddings(self):
        if False:
            i = 10
            return i + 15
        self.pad = [1, 1]
        self.padding_algorithm = 'VALID'

class TestConv2DOp_Valid_NHWC_MKLDNN(TestConv2DOp_Valid_MKLDNN):

    def init_data_format(self):
        if False:
            print('Hello World!')
        self.data_format = 'NHWC'

    def init_test_case_2(self):
        if False:
            return 10
        (N, C, H, W) = self.input_size
        self.input_size = [N, H, W, C]

class TestConv2DOp_Same_NHWC_MKLDNN(TestConv2DOp_Valid_NHWC_MKLDNN):

    def init_paddings(self):
        if False:
            return 10
        self.pad = [0, 0]
        self.padding_algorithm = 'SAME'

class TestConv2DOp_AsyPadding_NHWC_MKLDNN(TestConv2DOp_Valid_NHWC_MKLDNN):

    def init_paddings(self):
        if False:
            i = 10
            return i + 15
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestMKLDNNDilations(TestConv2DMKLDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        if False:
            print('Hello World!')
        self.dilations = [2, 2]

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()