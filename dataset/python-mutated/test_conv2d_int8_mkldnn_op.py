import os
import unittest
import numpy as np
from op_test import OpTest
from test_conv2d_op import TestConv2DOp, conv2d_forward_naive
from paddle.base import core

def conv2d_forward_refer(input, filter, group, conv_param):
    if False:
        i = 10
        return i + 15
    (out, _, _, _, _) = conv2d_forward_naive(input, filter, group, conv_param)
    return out

@unittest.skipIf(not core.supports_int8(), 'place does not support int8 computation')
class TestConv2DInt8Op(TestConv2DOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'conv2d'
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.data_format = 'NCHW'
        self.mkldnn_data_type = 'int8'
        self.weighttype = np.float32
        self.use_mkldnn = True
        self.init_weight_quantization_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_fuse_activation()
        self.init_fuse_residual()
        self.init_data_type()
        conv2d_param = {'stride': self.stride, 'pad': self.pad, 'dilation': self.dilations}
        inner_scale = 1.0 if self.fuse_activation != '' else self.scale_out
        activation_scale = self.scale_out if self.fuse_activation != '' else 1.0
        scale_output_shift = inner_scale / (self.scale_in * self.scale_weights[0])
        filter = np.random.random(self.filter_size).astype(self.weighttype)
        avx_scale = 0.5 if not core.supports_vnni() and self.srctype == np.int8 else 1.0
        filter_int = np.round(filter * self.scale_weights[0] * avx_scale).astype(np.int32)
        scale_output_shift = scale_output_shift / avx_scale

        def conv2d_forward_refer_helper(input_):
            if False:
                return 10
            return conv2d_forward_refer(input_.astype(np.int32), filter_int, self.groups, conv2d_param).astype(np.float32) * scale_output_shift

        def residual_helper(init_low, init_high, output_):
            if False:
                for i in range(10):
                    print('nop')
            input_residual_ = np.random.randint(init_low, init_high, self.input_residual_size).astype(self.srctype)
            return (output_ + input_residual_ * (inner_scale / self.scale_in_eltwise), input_residual_)
        if self.srctype == np.int8:
            (init_low, init_high) = (-5, 5)
            input = np.random.randint(init_low, init_high, self.input_size).astype(self.srctype)
            input_shift = (np.ones(self.input_size) * 128).astype(np.uint8)
            output1 = conv2d_forward_refer_helper(np.round(input + input_shift).astype(np.int32))
            output2 = conv2d_forward_refer_helper(np.round(input_shift).astype(np.int32))
            output = output1 - output2
        else:
            (init_low, init_high) = (0, 10)
            input = np.random.randint(init_low, init_high, self.input_size).astype(self.srctype)
            output = conv2d_forward_refer_helper(input)
        if self.fuse_residual:
            (output, input_residual) = residual_helper(init_low, init_high, output)
        if self.fuse_activation == '':
            pass
        elif self.fuse_activation == 'relu':
            output = activation_scale * np.maximum(output, 0)
        elif self.fuse_activation == 'hard_swish':
            output = activation_scale * output / 6.0 * np.minimum(np.maximum(0, output + 3.0), 6)
        elif self.fuse_activation == 'relu6':
            output = activation_scale * np.maximum(0, np.minimum(6, output))
        elif self.fuse_activation == 'swish':
            output = activation_scale * output / (1.0 + np.exp(-1.0 * output))
        elif self.fuse_activation == 'leaky_relu':
            output = activation_scale * np.maximum(output, 0.02 * output)
        else:
            raise NotImplementedError('test for ' + self.fuse_activation + ' activation not implemented')
        output = np.round(output).astype(self.dsttype)
        self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(input.astype(self.srctype)), 'Filter': OpTest.np_dtype_to_base_dtype(filter)}
        if self.fuse_residual:
            self.inputs['ResidualData'] = OpTest.np_dtype_to_base_dtype(input_residual)
        if self.fuse_activation != '' or self.fuse_residual:
            self.op_type = 'fused_conv2d'
        self.attrs = {'strides': self.stride, 'paddings': self.pad, 'groups': self.groups, 'dilations': self.dilations, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn, 'data_format': self.data_format, 'exhaustive_search': self.exhaustive_search, 'Scale_in': self.scale_in, 'Scale_out': self.scale_out, 'Scale_weights': self.scale_weights, 'Scale_in_eltwise': self.scale_in_eltwise, 'fuse_activation': self.fuse_activation, 'fuse_alpha': self.fuse_alpha, 'fuse_beta': self.fuse_beta, 'fuse_residual_connection': self.fuse_residual, 'mkldnn_data_type': self.mkldnn_data_type}
        self.outputs = {'Output': output}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), atol=1, check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        pass

    def test_check_grad_no_filter(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_no_input(self):
        if False:
            i = 10
            return i + 15
        pass

    def init_test_case(self):
        if False:
            print('Hello World!')
        TestConv2DOp.init_test_case(self)
        self.input_size = [1, 1, 5, 5]
        f_c = self.input_size[1] // self.groups
        self.input_residual_size = [1, 2, 3, 3]
        self.filter_size = [2, f_c, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [10.0] * self.filter_size[0] if self.per_channel_quantize_weight else [10.0]
        self.scale_in_eltwise = 0.6

    def init_weight_quantization_type(self):
        if False:
            print('Hello World!')
        self.per_channel_quantize_weight = False

    def init_data_type(self):
        if False:
            while True:
                i = 10
        self.srctype = np.uint8
        self.dsttype = np.int8

    def init_fuse_activation(self):
        if False:
            return 10
        self.fuse_activation = 'relu'
        self.fuse_alpha = 0
        self.fuse_beta = 0

    def init_fuse_residual(self):
        if False:
            while True:
                i = 10
        self.fuse_residual = True

class TestConv2D(TestConv2DInt8Op):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.6

class TestWithHardSwish(TestConv2D):

    def init_fuse_activation(self):
        if False:
            i = 10
            return i + 15
        self.fuse_activation = 'hard_swish'
        self.fuse_alpha = 1.0 / 6.0
        self.fuse_beta = 1.0 / 2.0

class TestWithRelu6(TestConv2D):

    def init_fuse_activation(self):
        if False:
            i = 10
            return i + 15
        self.fuse_activation = 'relu6'
        self.fuse_alpha = 0
        self.fuse_beta = 6

class TestWithSwish(TestConv2D):

    def init_fuse_activation(self):
        if False:
            for i in range(10):
                print('nop')
        self.fuse_activation = 'swish'
        self.fuse_alpha = 1
        self.fuse_beta = 0

class TestWithLeakyRelu(TestConv2D):

    def init_fuse_activation(self):
        if False:
            print('Hello World!')
        self.fuse_activation = 'leaky_relu'
        self.fuse_alpha = 0.02
        self.fuse_beta = 0

class TestWithPad(TestConv2D):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        TestConv2D.init_test_case(self)
        self.pad = [1, 1]
        self.input_residual_size = [2, 6, 5, 5]

class TestWithGroup(TestConv2D):

    def init_group(self):
        if False:
            print('Hello World!')
        self.groups = 3

class TestWithStride(TestConv2DInt8Op):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.8
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.5

class TestWithDilations(TestConv2DInt8Op):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [2, 2]
        self.input_size = [2, 3, 10, 10]
        self.input_residual_size = [2, 6, 8, 8]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.8
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.5

class TestWith1x1(TestConv2DInt8Op):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        self.input_residual_size = [1, 6, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [12.0]
        self.scale_in_eltwise = 0.5

class TestWithInput1x1Filter1x1(TestConv2DInt8Op):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        self.input_residual_size = [2, 6, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.8

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3

def init_data_type_with_fusion(self, input_dt, fuse_activation, fuse_residual):
    if False:
        for i in range(10):
            print('nop')
    self.op_type = 'fused_conv2d'
    self.srctype = input_dt
    self.dsttype = np.uint8 if fuse_activation == 'relu' else np.int8
    self.fuse_activation = fuse_activation
    self.fuse_residual = fuse_residual

class TestDepthwiseConv2d(TestConv2D):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [1, 32, 112, 112]
        self.input_residual_size = [1, 32, 112, 112]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [32, f_c, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [10.0] * self.filter_size[0] if self.per_channel_quantize_weight else [10.0]
        self.scale_in_eltwise = 0.8

    def init_group(self):
        if False:
            i = 10
            return i + 15
        self.groups = 32

    def init_weight_quantization_type(self):
        if False:
            return 10
        self.per_channel_quantize_weight = True

    def init_fuse_residual(self):
        if False:
            return 10
        self.fuse_residual = False

def create_test_int8_class(parent):
    if False:
        return 10

    class TestS8U8Case(parent):

        def init_data_type(self):
            if False:
                print('Hello World!')
            init_data_type_with_fusion(self, np.int8, 'relu', False)

    class TestS8S8Case(parent):

        def init_data_type(self):
            if False:
                i = 10
                return i + 15
            init_data_type_with_fusion(self, np.int8, '', False)

    class TestU8S8Case(parent):

        def init_data_type(self):
            if False:
                while True:
                    i = 10
            init_data_type_with_fusion(self, np.uint8, '', False)

    class TestU8U8Case(parent):

        def init_data_type(self):
            if False:
                for i in range(10):
                    print('nop')
            init_data_type_with_fusion(self, np.uint8, 'relu', False)

    class TestS8S8ResCase(parent):

        def init_data_type(self):
            if False:
                i = 10
                return i + 15
            init_data_type_with_fusion(self, np.int8, '', True)

    class TestU8S8ResCase(parent):

        def init_data_type(self):
            if False:
                return 10
            init_data_type_with_fusion(self, np.uint8, '', True)
    cls_name_s8u8 = '{}_relu_{}_residual_0'.format(parent.__name__, '1')
    cls_name_s8s8 = '{}_relu_{}_residual_0'.format(parent.__name__, '0')
    cls_name_u8s8 = '{}_relu_{}_residual_0'.format(parent.__name__, '0')
    cls_name_u8u8 = '{}_relu_{}_residual_0'.format(parent.__name__, '1')
    cls_name_s8s8_re_1 = '{}_relu_{}_residual_{}'.format(parent.__name__, '0', '1')
    cls_name_u8s8_re_1 = '{}_relu_{}_residual_{}'.format(parent.__name__, '0', '1')
    TestS8U8Case.__name__ = cls_name_s8u8
    TestS8S8Case.__name__ = cls_name_s8s8
    TestU8S8Case.__name__ = cls_name_u8s8
    TestU8U8Case.__name__ = cls_name_u8u8
    TestS8S8ResCase.__name__ = cls_name_s8s8_re_1
    TestU8S8ResCase.__name__ = cls_name_u8s8_re_1
    globals()[cls_name_s8u8] = TestS8U8Case
    globals()[cls_name_s8s8] = TestS8S8Case
    globals()[cls_name_u8s8] = TestU8S8Case
    globals()[cls_name_u8u8] = TestU8U8Case
    globals()[cls_name_s8s8_re_1] = TestS8S8ResCase
    globals()[cls_name_u8s8_re_1] = TestU8S8ResCase
    if os.name != 'nt':

        class TestS8U8ResCase(parent):

            def init_data_type(self):
                if False:
                    for i in range(10):
                        print('nop')
                init_data_type_with_fusion(self, np.int8, 'relu', True)
        cls_name_s8u8_re_1 = '{}_relu_{}_residual_{}'.format(parent.__name__, '1', '1')
        TestS8U8ResCase.__name__ = cls_name_s8u8_re_1
        globals()[cls_name_s8u8_re_1] = TestS8U8ResCase
create_test_int8_class(TestConv2DInt8Op)
create_test_int8_class(TestWithPad)
create_test_int8_class(TestWithStride)
create_test_int8_class(TestWithDilations)
create_test_int8_class(TestWithGroup)
create_test_int8_class(TestWith1x1)
create_test_int8_class(TestWithInput1x1Filter1x1)

class TestConv2DOp_AsyPadding_INT_MKLDNN(TestConv2DInt8Op):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True

    def init_paddings(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestConv2DOp_Same_INT_MKLDNN(TestConv2DOp_AsyPadding_INT_MKLDNN):

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0]
        self.padding_algorithm = 'SAME'

class TestConv2DOp_Valid_INT_MKLDNN(TestConv2DOp_AsyPadding_INT_MKLDNN):

    def init_paddings(self):
        if False:
            print('Hello World!')
        self.pad = [1, 1]
        self.padding_algorithm = 'VALID'
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()