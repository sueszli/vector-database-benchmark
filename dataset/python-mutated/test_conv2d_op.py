import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, get_numeric_gradient
from testsuite import create_op
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

def conv2d_forward_naive(input, filter, group, conv_param, padding_algorithm='EXPLICIT', data_format='NCHW'):
    if False:
        print('Hello World!')
    if padding_algorithm not in ['SAME', 'VALID', 'EXPLICIT']:
        raise ValueError("Unknown Attr(padding_algorithm): '%s'. It can only be 'SAME' or 'VALID'." % str(padding_algorithm))
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError("Unknown Attr(data_format): '%s' .It can only be 'NCHW' or 'NHWC'." % str(data_format))
    channel_last = data_format == 'NHWC'
    if channel_last:
        input = np.transpose(input, [0, 3, 1, 2])
    (in_n, in_c, in_h, in_w) = input.shape
    (f_n, f_c, f_h, f_w) = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group
    (stride, pad, dilation) = (conv_param['stride'], conv_param['pad'], conv_param['dilation'])

    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        if False:
            return 10
        padding = []
        for (input_size, filter_size, stride_size) in zip(input_shape, pool_size, pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(((out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding
    ksize = filter.shape[2:4]
    if padding_algorithm == 'VALID':
        pad = [0, 0, 0, 0]
    elif padding_algorithm == 'SAME':
        dilation = [1, 1]
        input_data_shape = input.shape[2:4]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)
    (pad_h_0, pad_h_1) = (pad[0], pad[0])
    (pad_w_0, pad_w_1) = (pad[1], pad[1])
    if len(pad) == 4:
        (pad_h_0, pad_h_1) = (pad[0], pad[1])
        (pad_w_0, pad_w_1) = (pad[2], pad[3])
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[0] * (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[1] * (f_w - 1) + 1)) // stride[1]
    out = np.zeros((out_n, out_c, out_h, out_w))
    d_bolck_h = dilation[0] * (f_h - 1) + 1
    d_bolck_w = dilation[1] * (f_w - 1) + 1
    input_pad = np.pad(input, ((0, 0), (0, 0), (pad_h_0, pad_h_1), (pad_w_0, pad_w_1)), mode='constant', constant_values=0)
    filter_dilation = np.zeros((f_n, f_c, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_h:dilation[0], 0:d_bolck_w:dilation[1]] = filter
    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = input_pad[:, g * f_c:(g + 1) * f_c, i * stride[0]:i * stride[0] + d_bolck_h, j * stride[1]:j * stride[1] + d_bolck_w]
                f_sub = filter_dilation[g * sub_f_n:(g + 1) * sub_f_n, :, :, :]
                for k in range(sub_out_c):
                    out[:, g * sub_out_c + k, i, j] = np.sum(input_pad_masked * f_sub[k, :, :, :], axis=(1, 2, 3))
    if channel_last:
        out = np.transpose(out, [0, 2, 3, 1])
    return (out, in_n, out_h, out_w, out_c)

def create_test_cudnn_class(parent):
    if False:
        while True:
            i = 10

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestCUDNNCase(parent):

        def init_kernel_type(self):
            if False:
                print('Hello World!')
            self.use_cudnn = True
            self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
    cls_name = '{}_{}'.format(parent.__name__, 'CUDNN')
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase

def create_test_cudnn_fp16_class(parent, grad_check=True):
    if False:
        while True:
            i = 10

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestConv2DCUDNNFp16(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            if False:
                return 10
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=0.02)

        def test_check_grad_no_filter(self):
            if False:
                print('Hello World!')
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                print('Hello World!')
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'})
    cls_name = '{}_{}'.format(parent.__name__, 'CUDNNFp16')
    TestConv2DCUDNNFp16.__name__ = cls_name
    globals()[cls_name] = TestConv2DCUDNNFp16

def create_test_cudnn_bf16_class(parent):
    if False:
        i = 10
        return i + 15

    @unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and do not support bfloat16')
    class TestConv2DCUDNNBF16(parent):

        def get_numeric_grad(self, place, check_name):
            if False:
                i = 10
                return i + 15
            scope = core.Scope()
            self._check_grad_helper()
            op = create_op(scope, self.op_type, self.inputs, self.outputs, self.attrs)
            return get_numeric_gradient(place, scope, op, self.inputs_fp32, check_name, ['Output'])

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            self.use_cudnn = True
            self.no_need_check_grad = True
            self.dtype = np.uint16

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=0.01)

        def test_check_grad_no_filter(self):
            if False:
                while True:
                    i = 10
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Input')
            self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'}, user_defined_grads=[numeric_grads])

        def test_check_grad_no_input(self):
            if False:
                for i in range(10):
                    print('nop')
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Filter')
            self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'}, user_defined_grads=[numeric_grads])
    cls_name = '{}_{}'.format(parent.__name__, 'CUDNNBF16')
    TestConv2DCUDNNBF16.__name__ = cls_name
    globals()[cls_name] = TestConv2DCUDNNBF16

def create_test_channel_last_class(parent):
    if False:
        while True:
            i = 10

    class TestChannelLastCase(parent):

        def init_data_format(self):
            if False:
                while True:
                    i = 10
            self.data_format = 'NHWC'

        def init_test_case_2(self):
            if False:
                return 10
            (N, C, H, W) = self.input_size
            self.input_size = [N, H, W, C]
    cls_name = '{}_{}'.format(parent.__name__, 'ChannelLast')
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase

def create_test_cudnn_channel_last_class(parent):
    if False:
        for i in range(10):
            print('nop')

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestCudnnChannelLastCase(parent):

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            self.use_cudnn = True
            self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64

        def init_data_format(self):
            if False:
                for i in range(10):
                    print('nop')
            self.data_format = 'NHWC'

        def init_test_case_2(self):
            if False:
                while True:
                    i = 10
            (N, C, H, W) = self.input_size
            self.input_size = [N, H, W, C]
    cls_name = '{}_{}'.format(parent.__name__, 'CudnnChannelLast')
    TestCudnnChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestCudnnChannelLastCase

def create_test_cudnn_channel_last_fp16_class(parent, grad_check=True):
    if False:
        for i in range(10):
            print('nop')

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestCudnnChannelLastFp16(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=0.02)

        def test_check_grad_no_filter(self):
            if False:
                while True:
                    i = 10
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                while True:
                    i = 10
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'})

        def init_data_format(self):
            if False:
                for i in range(10):
                    print('nop')
            self.data_format = 'NHWC'

        def init_test_case_2(self):
            if False:
                return 10
            (N, C, H, W) = self.input_size
            self.input_size = [N, H, W, C]
    cls_name = '{}_{}'.format(parent.__name__, 'CudnnChannelLastFp16')
    TestCudnnChannelLastFp16.__name__ = cls_name
    globals()[cls_name] = TestCudnnChannelLastFp16

def create_test_padding_SAME_class(parent):
    if False:
        while True:
            i = 10

    class TestPaddingSMAECase(parent):

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [0, 0]
            self.padding_algorithm = 'SAME'
    cls_name = '{}_{}'.format(parent.__name__, 'PaddingSAMEOp')
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase

def create_test_padding_VALID_class(parent):
    if False:
        return 10

    class TestPaddingVALIDCase(parent):

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1]
            self.padding_algorithm = 'VALID'
    cls_name = '{}_{}'.format(parent.__name__, 'PaddingVALIDOp')
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase

def create_test_cudnn_padding_SAME_class(parent):
    if False:
        i = 10
        return i + 15

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestCUDNNPaddingSMAECase(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cudnn = True
            self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1]
            self.padding_algorithm = 'SAME'
    cls_name = '{}_{}'.format(parent.__name__, 'CudnnPaddingSAMEOp')
    TestCUDNNPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingSMAECase

def create_test_cudnn_padding_VALID_class(parent):
    if False:
        i = 10
        return i + 15

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestCUDNNPaddingVALIDCase(parent):

        def init_kernel_type(self):
            if False:
                return 10
            self.use_cudnn = True
            self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1]
            self.padding_algorithm = 'VALID'
    cls_name = '{}_{}'.format(parent.__name__, 'CudnnPaddingVALIDOp')
    TestCUDNNPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingVALIDCase

def conv2d_wrapper(x, weight, stride=1, padding=0, padding_algorithm='EXPLICIT', dilation=1, groups=1, data_format='NCDHW'):
    if False:
        while True:
            i = 10
    if data_format == 'AnyLayout':
        data_format = 'NCDHW'
    if padding_algorithm is None:
        padding_algorithm = 'EXPLICIT'
    return paddle._C_ops.conv2d(x, weight, stride, padding, padding_algorithm, dilation, groups, data_format)

class TestConv2DOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'conv2d'
        self.python_api = conv2d_wrapper
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.data_format = 'AnyLayout'
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        conv2d_param = {'stride': self.stride, 'pad': self.pad, 'dilation': self.dilations}
        if self.is_bfloat16_op():
            input = np.random.random(self.input_size).astype(np.float32)
            filter = np.random.uniform(-1, 1, self.filter_size).astype(np.float32)
        else:
            input = np.random.random(self.input_size).astype(self.dtype)
            filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        (output, _, _, _, _) = conv2d_forward_naive(input2, filter, self.groups, conv2d_param)
        if self.is_bfloat16_op():
            output = output.astype(np.float32)
            self.inputs = {'Input': convert_float_to_uint16(input), 'Filter': convert_float_to_uint16(filter)}
            self.inputs_fp32 = {'Input': OpTest.np_dtype_to_base_dtype(input), 'Filter': OpTest.np_dtype_to_base_dtype(filter)}
        else:
            output = output.astype(self.dtype)
            self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(input), 'Filter': OpTest.np_dtype_to_base_dtype(filter)}
        self.attrs = {'strides': self.stride, 'paddings': self.pad, 'groups': self.groups, 'dilations': self.dilations, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn, 'data_format': self.data_format, 'fuse_relu_before_depthwise_conv': self.fuse_relu_before_depthwise_conv, 'exhaustive_search': self.exhaustive_search}
        self.outputs = {'Output': output}

    def has_cuda(self):
        if False:
            return 10
        return core.is_compiled_with_cuda() and (self.use_cudnn or self.use_cuda)

    def test_check_output(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_output_with_place(place, atol=1e-05, check_dygraph=not self.use_mkldnn)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        if self.dtype == np.float16 or (hasattr(self, 'no_need_check_grad') and self.no_need_check_grad):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output', max_relative_error=0.02, check_dygraph=not self.use_mkldnn)

    def test_check_grad_no_filter(self):
        if False:
            return 10
        if self.dtype == np.float16 or (hasattr(self, 'no_need_check_grad') and self.no_need_check_grad):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, ['Input'], 'Output', max_relative_error=0.02, no_grad_set={'Filter'}, check_dygraph=not self.use_mkldnn)

    def test_check_grad_no_input(self):
        if False:
            print('Hello World!')
        if self.dtype == np.float16 or (hasattr(self, 'no_need_check_grad') and self.no_need_check_grad):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'}, check_dygraph=not self.use_mkldnn)

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_test_case_2(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def init_dilation(self):
        if False:
            for i in range(10):
                print('nop')
        self.dilations = [1, 1]

    def init_group(self):
        if False:
            i = 10
            return i + 15
        self.groups = 1

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        pass

class TestWithPad(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

class TestWithStride(TestConv2DOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

class TestWithGroup(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [18, f_c, 3, 3]

class TestWith1x1(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3

class TestWithDepthWise3x3(TestConv2DOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [3, 4, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        if False:
            while True:
                i = 10
        self.dilations = [2, 2]

    def init_group(self):
        if False:
            print('Hello World!')
        self.groups = 4

class TestWithDepthWise5x5(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 4, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [8, f_c, 5, 5]

    def init_group(self):
        if False:
            return 10
        self.groups = 4

class TestWithDepthWise7x7(TestConv2DOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 8, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]

    def init_group(self):
        if False:
            print('Hello World!')
        self.groups = 8

class TestWithDilation(TestConv2DOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        if False:
            return 10
        self.dilations = [2, 2]

    def init_group(self):
        if False:
            i = 10
            return i + 15
        self.groups = 3

class TestWithInput1x1Filter1x1(TestConv2DOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [100, 3, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        if False:
            i = 10
            return i + 15
        self.groups = 3
create_test_cudnn_class(TestConv2DOp)
create_test_cudnn_class(TestWithPad)
create_test_cudnn_class(TestWithStride)
create_test_cudnn_class(TestWithGroup)
create_test_cudnn_class(TestWith1x1)
create_test_cudnn_class(TestWithInput1x1Filter1x1)
create_test_cudnn_fp16_class(TestConv2DOp)
create_test_cudnn_fp16_class(TestWithPad)
create_test_cudnn_fp16_class(TestWithStride)
create_test_cudnn_fp16_class(TestWithGroup)
create_test_cudnn_fp16_class(TestWith1x1)
create_test_cudnn_fp16_class(TestWithInput1x1Filter1x1)
create_test_cudnn_bf16_class(TestConv2DOp)
create_test_cudnn_bf16_class(TestWithPad)
create_test_cudnn_bf16_class(TestWithStride)
create_test_cudnn_bf16_class(TestWithGroup)
create_test_cudnn_bf16_class(TestWith1x1)
create_test_cudnn_bf16_class(TestWithInput1x1Filter1x1)

class TestCUDNNExhaustiveSearch(TestConv2DOp):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_cudnn = True
        self.exhaustive_search = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64

class TestConv2DOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        with program_guard(Program(), Program()):

            def test_Variable():
                if False:
                    print('Hello World!')
                x1 = base.create_lod_tensor(np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace())
                paddle.static.nn.conv2d(x1, 1, 1)
            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                if False:
                    for i in range(10):
                        print('nop')
                x2 = paddle.static.data(name='x2', shape=[-1, 3, 4, 5, 6], dtype='int32')
                paddle.static.nn.conv2d(x2, 1, 1)
            self.assertRaises(TypeError, test_dtype)

class TestConv2DOp_v2(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'conv2d'
        self.python_api = conv2d_wrapper
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_data_format()
        self.init_test_case()
        self.init_paddings()
        self.init_test_case_2()
        conv2d_param = {'stride': self.stride, 'pad': self.pad, 'dilation': self.dilations}
        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)
        (output, _, _, _, _) = conv2d_forward_naive(input2, filter, self.groups, conv2d_param, self.padding_algorithm, self.data_format)
        output = output.astype(self.dtype)
        self.inputs = {'Input': OpTest.np_dtype_to_base_dtype(input), 'Filter': OpTest.np_dtype_to_base_dtype(filter)}
        self.attrs = {'strides': self.stride, 'paddings': self.pad, 'padding_algorithm': self.padding_algorithm, 'groups': self.groups, 'dilations': self.dilations, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn, 'data_format': self.data_format, 'fuse_relu_before_depthwise_conv': self.fuse_relu_before_depthwise_conv, 'exhaustive_search': self.exhaustive_search}
        self.outputs = {'Output': output}

    def has_cuda(self):
        if False:
            return 10
        return core.is_compiled_with_cuda() and (self.use_cudnn or self.use_cuda)

    def test_check_output(self):
        if False:
            return 10
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_output_with_place(place, atol=1e-05, check_dygraph=not self.use_mkldnn)

    def test_check_grad(self):
        if False:
            return 10
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output', max_relative_error=0.02, check_dygraph=not self.use_mkldnn)

    def test_check_grad_no_filter(self):
        if False:
            while True:
                i = 10
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, ['Input'], 'Output', max_relative_error=0.02, no_grad_set={'Filter'}, check_dygraph=not self.use_mkldnn)

    def test_check_grad_no_input(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'}, check_dygraph=not self.use_mkldnn)

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 4, 3]

    def init_dilation(self):
        if False:
            i = 10
            return i + 15
        self.dilations = [1, 1]

    def init_group(self):
        if False:
            i = 10
            return i + 15
        self.groups = 1

    def init_kernel_type(self):
        if False:
            return 10
        pass

    def init_paddings(self):
        if False:
            i = 10
            return i + 15
        self.pad = [0, 0]
        self.padding_algorithm = 'EXPLICIT'

    def init_data_format(self):
        if False:
            for i in range(10):
                print('nop')
        self.data_format = 'NCHW'

    def init_test_case_2(self):
        if False:
            print('Hello World!')
        pass

class TestConv2DOp_AsyPadding(TestConv2DOp_v2):

    def init_paddings(self):
        if False:
            return 10
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestWithPad_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        if False:
            i = 10
            return i + 15
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestWithStride_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            return 10
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        if False:
            i = 10
            return i + 15
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestWithGroup_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 4, 3]

class TestWith1x1_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3

    def init_paddings(self):
        if False:
            print('Hello World!')
        self.pad = [2, 2, 4, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestWithDepthWise3x3_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.stride = [1, 1]
        self.input_size = [3, 4, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 3, 3]

    def init_dilation(self):
        if False:
            i = 10
            return i + 15
        self.dilations = [2, 2]

    def init_group(self):
        if False:
            return 10
        self.groups = 4

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = 'EXPLICIT'

class TestWithDepthWise5x5_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.stride = [1, 1]
        self.input_size = [2, 4, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [8, f_c, 5, 5]

    def init_group(self):
        if False:
            return 10
        self.groups = 4

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [0, 1, 1, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestWithDepthWise7x7_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.stride = [2, 2]
        self.input_size = [2, 8, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]

    def init_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.groups = 8

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 3, 4, 1]
        self.padding_algorithm = 'EXPLICIT'

class TestWithDilation_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_dilation(self):
        if False:
            print('Hello World!')
        self.dilations = [2, 2]

    def init_group(self):
        if False:
            print('Hello World!')
        self.groups = 3

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.pad = [0, 1, 3, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestWithInput1x1Filter1x1_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.stride = [1, 1]
        self.input_size = [40, 3, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        if False:
            while True:
                i = 10
        self.groups = 3

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [0, 3, 4, 0]
        self.padding_algorithm = 'EXPLICIT'
create_test_cudnn_class(TestConv2DOp_AsyPadding)
create_test_cudnn_class(TestWithPad_AsyPadding)
create_test_cudnn_class(TestWithStride_AsyPadding)
create_test_cudnn_class(TestWithGroup_AsyPadding)
create_test_cudnn_class(TestWith1x1_AsyPadding)
create_test_cudnn_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_padding_SAME_class(TestConv2DOp_AsyPadding)
create_test_padding_SAME_class(TestWithPad_AsyPadding)
create_test_padding_SAME_class(TestWithStride_AsyPadding)
create_test_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_padding_VALID_class(TestConv2DOp_AsyPadding)
create_test_padding_VALID_class(TestWithPad_AsyPadding)
create_test_padding_VALID_class(TestWithStride_AsyPadding)
create_test_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_cudnn_padding_SAME_class(TestConv2DOp_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithPad_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithStride_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_cudnn_padding_VALID_class(TestConv2DOp_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithPad_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithStride_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_channel_last_class(TestConv2DOp_AsyPadding)
create_test_channel_last_class(TestWithPad_AsyPadding)
create_test_channel_last_class(TestWithGroup_AsyPadding)
create_test_channel_last_class(TestWith1x1_AsyPadding)
create_test_channel_last_class(TestWithInput1x1Filter1x1_AsyPadding)
create_test_cudnn_channel_last_class(TestConv2DOp_AsyPadding)
create_test_cudnn_channel_last_class(TestWithPad_AsyPadding)
create_test_cudnn_channel_last_class(TestWithStride_AsyPadding)
create_test_cudnn_channel_last_class(TestWithGroup_AsyPadding)
create_test_cudnn_channel_last_class(TestWithDilation_AsyPadding)
create_test_cudnn_channel_last_fp16_class(TestConv2DOp_AsyPadding)
create_test_cudnn_channel_last_fp16_class(TestWithPad_AsyPadding)
create_test_cudnn_channel_last_fp16_class(TestWithStride_AsyPadding)
create_test_cudnn_channel_last_fp16_class(TestWithGroup_AsyPadding)
create_test_cudnn_channel_last_fp16_class(TestWithDilation_AsyPadding)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()