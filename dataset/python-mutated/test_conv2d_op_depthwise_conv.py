import unittest
import numpy as np
import paddle
paddle.enable_static()
from op_test import get_numeric_gradient
from test_conv2d_op import TestConv2DOp, TestConv2DOp_v2, create_test_channel_last_class, create_test_cudnn_channel_last_class, create_test_cudnn_padding_SAME_class, create_test_padding_SAME_class, create_test_padding_VALID_class
from testsuite import create_op
from paddle.base import core

def depthwise_conv2d_wrapper(x, weight, stride=1, padding=0, padding_algorithm='EXPLICIT', groups=1, dilation=1, data_format='NCDHW'):
    if False:
        return 10
    if data_format == 'AnyLayout':
        data_format = 'NCDHW'
    if padding_algorithm is None:
        padding_algorithm = 'EXPLICIT'
    return paddle._C_ops.depthwise_conv2d(x, weight, stride, padding, padding_algorithm, groups, dilation, data_format)

class TestDepthwiseConv(TestConv2DOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConv2(TestConv2DOp):

    def init_test_case(self):
        if False:
            return 10
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConv3(TestConv2DOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConvWithDilation(TestConv2DOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConvWithDilation2(TestConv2DOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConvandFuse(TestConv2DOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConv2andFuse(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConv3andFuse(TestConv2DOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConvWithDilationandFuse(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConvWithDilation2andFuse(TestConv2DOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

class TestDepthwiseConv_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.use_cuda = True
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConv2_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.use_cuda = True
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConv3_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_cuda = True
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 1, 0, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConvWithDilation_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 1, 2, 1]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConvWithDilation2_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [0, 1, 1, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConvandFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConv2andFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 1, 1, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConv3andFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 2, 0, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConvWithDilationandFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            print('Hello World!')
        self.pad = [2, 1, 1, 0]
        self.padding_algorithm = 'EXPLICIT'

class TestDepthwiseConvWithDilation2andFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = 'depthwise_conv2d'
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 3, 1, 3]
        self.padding_algorithm = 'EXPLICIT'

def create_test_fp16_class(parent, grad_check=True):
    if False:
        for i in range(10):
            print('nop')

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestDepthwiseConvFP16(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cuda = True
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
                return 10
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                print('Hello World!')
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'})
    cls_name = '{}_{}'.format(parent.__name__, 'FP16OP')
    TestDepthwiseConvFP16.__name__ = cls_name
    globals()[cls_name] = TestDepthwiseConvFP16

def create_test_bf16_class(parent, atol=0.01):
    if False:
        while True:
            i = 10

    @unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and do not support bfloat16')
    class TestDepthwiseConvBF16(parent):

        def get_numeric_grad(self, place, check_name):
            if False:
                print('Hello World!')
            scope = core.Scope()
            self._check_grad_helper()
            op = create_op(scope, self.op_type, self.inputs, self.outputs, self.attrs)
            return get_numeric_gradient(place, scope, op, self.inputs_fp32, check_name, ['Output'])

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cuda = True
            self.no_need_check_grad = True
            self.dtype = np.uint16

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=atol)

        def test_check_grad_no_filter(self):
            if False:
                return 10
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Input')
            self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'}, user_defined_grads=[numeric_grads])

        def test_check_grad_no_input(self):
            if False:
                return 10
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Filter')
            self.check_grad_with_place(place, ['Filter'], 'Output', no_grad_set={'Input'}, user_defined_grads=[numeric_grads])
    cls_name = '{}_{}'.format(parent.__name__, 'BF16OP')
    TestDepthwiseConvBF16.__name__ = cls_name
    globals()[cls_name] = TestDepthwiseConvBF16

def create_test_channel_last_fp16_class(parent, grad_check=True):
    if False:
        while True:
            i = 10

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestChannelLastFP16(parent):

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            self.use_cuda = True
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
                for i in range(10):
                    print('nop')
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(place, ['Input'], 'Output', no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                return 10
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
                i = 10
                return i + 15
            (N, C, H, W) = self.input_size
            self.input_size = [N, H, W, C]
    cls_name = '{}_{}'.format(parent.__name__, 'ChannelLastFP16')
    TestChannelLastFP16.__name__ = cls_name
    globals()[cls_name] = TestChannelLastFP16
create_test_fp16_class(TestDepthwiseConv)
create_test_fp16_class(TestDepthwiseConv2)
create_test_fp16_class(TestDepthwiseConv3)
create_test_fp16_class(TestDepthwiseConvWithDilation)
create_test_fp16_class(TestDepthwiseConvWithDilation2)
create_test_fp16_class(TestDepthwiseConvandFuse)
create_test_fp16_class(TestDepthwiseConv2andFuse)
create_test_fp16_class(TestDepthwiseConv3andFuse)
create_test_fp16_class(TestDepthwiseConvWithDilationandFuse)
create_test_fp16_class(TestDepthwiseConvWithDilation2andFuse)
create_test_bf16_class(TestDepthwiseConv)
create_test_bf16_class(TestDepthwiseConv2)
create_test_bf16_class(TestDepthwiseConv3, atol=0.04)
create_test_bf16_class(TestDepthwiseConvWithDilation)
create_test_bf16_class(TestDepthwiseConvWithDilation2)
create_test_bf16_class(TestDepthwiseConvandFuse)
create_test_bf16_class(TestDepthwiseConv2andFuse)
create_test_bf16_class(TestDepthwiseConv3andFuse)
create_test_bf16_class(TestDepthwiseConvWithDilationandFuse)
create_test_bf16_class(TestDepthwiseConvWithDilation2andFuse)
create_test_padding_SAME_class(TestDepthwiseConv_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvWithDilation_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConv_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvWithDilation_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)
create_test_channel_last_class(TestDepthwiseConv_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvWithDilation2_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)
create_test_channel_last_fp16_class(TestDepthwiseConv_AsyPadding)
create_test_channel_last_fp16_class(TestDepthwiseConvWithDilation2_AsyPadding)
create_test_channel_last_fp16_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_channel_last_fp16_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)
if core.is_compiled_with_rocm():
    create_test_cudnn_padding_SAME_class(TestDepthwiseConv_AsyPadding)
    create_test_cudnn_padding_SAME_class(TestDepthwiseConvWithDilation_AsyPadding)
    create_test_padding_VALID_class(TestDepthwiseConv_AsyPadding)
    create_test_padding_VALID_class(TestDepthwiseConvWithDilation_AsyPadding)
    create_test_cudnn_channel_last_class(TestDepthwiseConv_AsyPadding)
    create_test_cudnn_channel_last_class(TestDepthwiseConvWithDilation2_AsyPadding)
if __name__ == '__main__':
    unittest.main()