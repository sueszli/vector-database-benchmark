import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class
from op_test_xpu import XPUOpTest
import paddle

def conv3d_forward_naive(input, filter, group, conv_param, padding_algorithm='EXPLICIT', data_format='NCDHW'):
    if False:
        i = 10
        return i + 15
    if padding_algorithm not in ['SAME', 'VALID', 'EXPLICIT']:
        raise ValueError("Unknown Attr(padding_algorithm): '%s'. It can only be 'SAME' or 'VALID'." % str(padding_algorithm))
    if data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError("Unknown Attr(data_format): '%s' .It can only be 'NCDHW' or 'NDHWC'." % str(data_format))
    channel_last = data_format == 'NDHWC'
    if channel_last:
        input = np.transpose(input, [0, 4, 1, 2, 3])
    (in_n, in_c, in_d, in_h, in_w) = input.shape
    (f_n, f_c, f_d, f_h, f_w) = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group
    (stride, pad, dilation) = (conv_param['stride'], conv_param['pad'], conv_param['dilations'])

    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        if False:
            i = 10
            return i + 15
        padding = []
        for (input_size, filter_size, stride_size) in zip(input_shape, pool_size, pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(((out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding
    ksize = filter.shape[2:5]
    if padding_algorithm == 'VALID':
        pad = [0, 0, 0, 0, 0, 0]
    elif padding_algorithm == 'SAME':
        dilation = [1, 1, 1]
        input_data_shape = input.shape[2:5]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)
    (pad_d_0, pad_d_1) = (pad[0], pad[0])
    (pad_h_0, pad_h_1) = (pad[1], pad[1])
    (pad_w_0, pad_w_1) = (pad[2], pad[2])
    if len(pad) == 6:
        (pad_d_0, pad_d_1) = (pad[0], pad[1])
        (pad_h_0, pad_h_1) = (pad[2], pad[3])
        (pad_w_0, pad_w_1) = (pad[4], pad[5])
    out_d = 1 + (in_d + pad_d_0 + pad_d_1 - (dilation[0] * (f_d - 1) + 1)) // stride[0]
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[1] * (f_h - 1) + 1)) // stride[1]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[2] * (f_w - 1) + 1)) // stride[2]
    out = np.zeros((in_n, out_c, out_d, out_h, out_w))
    d_bolck_d = dilation[0] * (f_d - 1) + 1
    d_bolck_h = dilation[1] * (f_h - 1) + 1
    d_bolck_w = dilation[2] * (f_w - 1) + 1
    input_pad = np.pad(input, ((0, 0), (0, 0), (pad_d_0, pad_d_1), (pad_h_0, pad_h_1), (pad_w_0, pad_w_1)), mode='constant', constant_values=0)
    filter_dilation = np.zeros((f_n, f_c, d_bolck_d, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_d:dilation[0], 0:d_bolck_h:dilation[1], 0:d_bolck_w:dilation[2]] = filter
    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                for g in range(group):
                    input_pad_masked = input_pad[:, g * f_c:(g + 1) * f_c, d * stride[0]:d * stride[0] + d_bolck_d, i * stride[1]:i * stride[1] + d_bolck_h, j * stride[2]:j * stride[2] + d_bolck_w]
                    f_sub = filter_dilation[g * sub_f_n:(g + 1) * sub_f_n, :, :, :, :]
                    for k in range(sub_out_c):
                        out[:, g * sub_out_c + k, d, i, j] = np.sum(input_pad_masked * f_sub[k, :, :, :, :], axis=(1, 2, 3, 4))
    if channel_last:
        out = np.transpose(out, [0, 2, 3, 4, 1])
    return out

def create_test_padding_SAME_class(parent):
    if False:
        i = 10
        return i + 15

    class TestPaddingSMAECase(parent):

        def init_paddings(self):
            if False:
                for i in range(10):
                    print('nop')
            self.pad = [0, 0, 0]
            self.padding_algorithm = 'SAME'
    cls_name = '{}_{}'.format(parent.__name__, 'PaddingSAMEOp')
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase

def create_test_padding_VALID_class(parent):
    if False:
        for i in range(10):
            print('nop')

    class TestPaddingVALIDCase(parent):

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1, 1]
            self.padding_algorithm = 'VALID'
    cls_name = '{}_{}'.format(parent.__name__, 'PaddingVALIDOp')
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase

def create_test_channel_last_class(parent):
    if False:
        while True:
            i = 10

    class TestChannelLastCase(parent):

        def init_data_format(self):
            if False:
                i = 10
                return i + 15
            self.data_format = 'NDHWC'

        def init_test_case_2(self):
            if False:
                while True:
                    i = 10
            (N, C, D, H, W) = self.input_size
            self.input_size = [N, D, H, W, C]
    cls_name = '{}_{}'.format(parent.__name__, 'ChannelLast')
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase
paddle.enable_static()

class XPUTestConv3DOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'conv3d'
        self.use_dynamic_create_class = False

    class TestConv3DOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type
            self.op_type = 'conv3d'
            self.use_cudnn = False
            self.use_mkldnn = False
            self.data_format = 'AnyLayout'
            self.init_kernel_type()
            self.init_group()
            self.init_dilation()
            self.init_test_case()
            conv3d_param = {'stride': self.stride, 'pad': self.pad, 'dilations': self.dilations}
            np.random.seed(100)
            input = np.random.random(self.input_size).astype(self.dtype)
            filter = np.random.random(self.filter_size).astype(self.dtype)
            output = conv3d_forward_naive(input, filter, self.groups, conv3d_param).astype(self.dtype)
            self.inputs = {'Input': XPUOpTest.np_dtype_to_base_dtype(input), 'Filter': XPUOpTest.np_dtype_to_base_dtype(filter)}
            self.attrs = {'strides': self.stride, 'paddings': self.pad, 'groups': self.groups, 'dilations': self.dilations, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn, 'data_format': self.data_format}
            self.outputs = {'Output': output}

        def test_check_output(self):
            if False:
                print('Hello World!')
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output', max_relative_error=0.03)

        def test_check_grad_no_filter(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Input'], 'Output', max_relative_error=0.03, no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Filter'], 'Output', max_relative_error=0.03, no_grad_set={'Input'})

        def init_test_case(self):
            if False:
                return 10
            self.pad = [0, 0, 0]
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

        def init_test_case_2(self):
            if False:
                print('Hello World!')
            pass

        def init_dilation(self):
            if False:
                return 10
            self.dilations = [1, 1, 1]

        def init_group(self):
            if False:
                while True:
                    i = 10
            self.groups = 1

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            pass

    class TestCase1(TestConv3DOp):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1, 1]
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

    class TestWithGroup1(TestConv3DOp):

        def init_group(self):
            if False:
                while True:
                    i = 10
            self.groups = 3

    class TestWithGroup2(TestCase1):

        def init_group(self):
            if False:
                i = 10
                return i + 15
            self.groups = 3

    class TestWith1x1(TestConv3DOp):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.pad = [0, 0, 0]
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [120, f_c, 1, 1, 1]

        def init_dilation(self):
            if False:
                return 10
            self.dilations = [1, 1, 1]

        def init_group(self):
            if False:
                for i in range(10):
                    print('nop')
            self.groups = 3

    class TestWithInput1x1Filter1x1(TestConv3DOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.pad = [0, 0, 0]
            self.stride = [1, 1, 1]
            self.input_size = [40, 3, 1, 1, 1]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [120, f_c, 1, 1, 1]

        def init_dilation(self):
            if False:
                i = 10
                return i + 15
            self.dilations = [1, 1, 1]

        def init_group(self):
            if False:
                return 10
            self.groups = 3

    class TestWithDilation(TestConv3DOp):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.pad = [0, 0, 0]
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 6, 6, 6]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 2, 2, 2]

        def init_dilation(self):
            if False:
                while True:
                    i = 10
            self.dilations = [2, 2, 2]

        def init_group(self):
            if False:
                return 10
            self.groups = 3

class XPUTestConv3DOp_v2(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'conv3d'
        self.use_dynamic_create_class = False

    class TestConv3DOp_2(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type
            self.op_type = 'conv3d'
            self.use_cudnn = False
            self.use_mkldnn = False
            self.data_format = 'NCDHW'
            self.init_kernel_type()
            self.init_group()
            self.init_dilation()
            self.init_data_format()
            self.init_test_case()
            self.init_paddings()
            self.init_test_case_2()
            conv3d_param = {'stride': self.stride, 'pad': self.pad, 'dilations': self.dilations}
            np.random.seed(100)
            input = np.random.random(self.input_size).astype(self.dtype)
            filter = np.random.random(self.filter_size).astype(self.dtype)
            output = conv3d_forward_naive(input, filter, self.groups, conv3d_param, self.padding_algorithm, self.data_format).astype(self.dtype)
            self.inputs = {'Input': XPUOpTest.np_dtype_to_base_dtype(input), 'Filter': XPUOpTest.np_dtype_to_base_dtype(filter)}
            self.attrs = {'strides': self.stride, 'paddings': self.pad, 'padding_algorithm': self.padding_algorithm, 'groups': self.groups, 'dilations': self.dilations, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn, 'data_format': self.data_format}
            self.outputs = {'Output': output}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output', max_relative_error=0.03)

        def test_check_grad_no_filter(self):
            if False:
                i = 10
                return i + 15
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Input'], 'Output', max_relative_error=0.03, no_grad_set={'Filter'})

        def test_check_grad_no_input(self):
            if False:
                print('Hello World!')
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Filter'], 'Output', max_relative_error=0.03, no_grad_set={'Input'})

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

        def init_test_case_2(self):
            if False:
                return 10
            pass

        def init_dilation(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dilations = [1, 1, 1]

        def init_group(self):
            if False:
                i = 10
                return i + 15
            self.groups = 1

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            pass

        def init_paddings(self):
            if False:
                i = 10
                return i + 15
            self.pad = [0, 0, 0]
            self.padding_algorithm = 'EXPLICIT'

        def init_data_format(self):
            if False:
                return 10
            self.data_format = 'NCDHW'

    class TestConv3DOp_AsyPadding(TestConv3DOp_2):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.stride = [1, 1, 2]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 0, 1, 0, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestConv3DOp_DiffDataInDiffDim(TestConv3DOp_2):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.stride = [1, 1, 2]
            self.input_size = [2, 3, 4, 5, 5]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 4, 3]

        def init_paddings(self):
            if False:
                i = 10
                return i + 15
            self.pad = [1, 0, 1, 0, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestCase1_AsyPadding(TestConv3DOp_2):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

        def init_paddings(self):
            if False:
                return 10
            self.pad = [0, 0, 1, 0, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestWithGroup1_AsyPadding(TestConv3DOp_2):

        def init_group(self):
            if False:
                while True:
                    i = 10
            self.groups = 3

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1, 1, 0, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestWithGroup2_AsyPadding(TestConv3DOp_2):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 4, 4, 4]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [6, f_c, 3, 3, 3]

        def init_group(self):
            if False:
                for i in range(10):
                    print('nop')
            self.groups = 3

        def init_paddings(self):
            if False:
                for i in range(10):
                    print('nop')
            self.pad = [1, 1, 0, 1, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestWithDilation_AsyPadding(TestConv3DOp_2):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.stride = [1, 1, 1]
            self.input_size = [2, 3, 6, 6, 6]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 2, 2, 2]

        def init_dilation(self):
            if False:
                return 10
            self.dilations = [2, 2, 2]

        def init_group(self):
            if False:
                i = 10
                return i + 15
            self.groups = 3

        def init_paddings(self):
            if False:
                while True:
                    i = 10
            self.pad = [0, 0, 1, 0, 1, 0]
            self.padding_algorithm = 'EXPLICIT'

class TestConv3DAPI(unittest.TestCase):

    def test_api(self):
        if False:
            i = 10
            return i + 15
        input_NDHWC = paddle.static.data(name='input_NDHWC', shape=[2, 5, 5, 5, 3], dtype='float32')
        input_NCDHW = paddle.static.data(name='input_NCDHW', shape=[2, 3, 5, 5, 3], dtype='float32')
        paddle.static.nn.conv3d(input=input_NDHWC, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding=0, dilation=[1, 1, 1], groups=1, data_format='NCDHW')
        paddle.static.nn.conv3d(input=input_NCDHW, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 2, 1, 0, 1, 0], dilation=[1, 1, 1], groups=1, data_format='NCDHW')
        paddle.static.nn.conv3d(input=input_NCDHW, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], dilation=[1, 1, 1], groups=1, data_format='NCDHW')
        paddle.static.nn.conv3d(input=input_NDHWC, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], dilation=[1, 1, 1], groups=1, data_format='NDHWC')
        paddle.static.nn.conv3d(input=input_NCDHW, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding='SAME', dilation=[1, 1, 1], groups=1, data_format='NCDHW')
        paddle.static.nn.conv3d(input=input_NCDHW, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding='VALID', dilation=[1, 1, 1], groups=1, data_format='NCDHW')

class TestConv3DAPI_Error(unittest.TestCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        input = paddle.static.data(name='input', shape=[2, 5, 5, 5, 4], dtype='float32')

        def run_1():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=3, stride=1, padding=0, dilation=1, groups=1, use_cudnn=[0], data_format='NCDHW')
        self.assertRaises(ValueError, run_1)

        def run_2():
            if False:
                i = 10
                return i + 15
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=[3, 3, 3], stride=[1, 1, 1], padding=0, dilation=[1, 1, 1], groups=1, use_cudnn=False, data_format='NCHWC')
        self.assertRaises(ValueError, run_2)

        def run_3():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=3, stride=1, padding='SAMEE', dilation=1, groups=1, use_cudnn=False, data_format='NCDHW')
        self.assertRaises(ValueError, run_3)

        def run_4():
            if False:
                print('Hello World!')
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=3, stride=1, padding=[[0, 1], [0, 0], [0, 1], [0, 1], [0, 1]], dilation=1, groups=1, use_cudnn=False, data_format='NCDHW')
        self.assertRaises(ValueError, run_4)

        def run_5():
            if False:
                print('Hello World!')
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=0, stride=0, padding=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dilation=1, groups=1, use_cudnn=False, data_format='NDHWC')
        self.assertRaises(ValueError, run_5)
        x = paddle.static.data(name='x', shape=[2, 5, 5, 5, -1], dtype='float32')

        def run_6():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv3d(input=x, num_filters=3, filter_size=3, stride=1, padding=0, dilation=1, groups=1, use_cudnn=False, data_format='NDHWC')
        self.assertRaises(ValueError, run_6)

        def run_7():
            if False:
                return 10
            paddle.static.nn.conv3d(input=input, num_filters=3, filter_size=3, stride=1, padding=0, dilation=1, groups=3, use_cudnn=False, data_format='NDHWC')
        self.assertRaises(ValueError, run_7)

        def run_8():
            if False:
                for i in range(10):
                    print('nop')
            paddle.static.nn.conv3d(input=input, num_filters=0, filter_size=0, stride=0, padding=0, dilation=0, groups=1, use_cudnn=False, data_format='NDHWC')
        self.assertRaises(ValueError, run_8)
for stype in ['float32']:
    create_test_class(globals(), XPUTestConv3DOp, stype)
    create_test_class(globals(), XPUTestConv3DOp_v2, stype)
if __name__ == '__main__':
    unittest.main()