import unittest
from unittest import TestCase
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base

class TestFunctionalConv3D(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
    dtype = 'float32'

    def setUp(self):
        if False:
            print('Hello World!')
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

    def prepare(self):
        if False:
            while True:
                i = 10
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape,) * 3
        else:
            filter_shape = tuple(self.filter_shape)
        self.weight = np.random.uniform(-1, 1, (self.out_channels, self.in_channels // self.groups) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (self.out_channels,)).astype(self.dtype)
        self.channel_last = self.data_format == 'NDHWC'
        if self.channel_last:
            self.input_shape = (self.batch_size,) + self.spatial_shape + (self.in_channels,)
        else:
            self.input_shape = (self.batch_size, self.in_channels) + self.spatial_shape
        self.input = np.random.uniform(-1, 1, self.input_shape).astype(self.dtype)

    def static_graph_case_1(self):
        if False:
            print('Hello World!')
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                if self.channel_last:
                    x = paddle.static.data('input', (-1, -1, -1, -1, self.in_channels), dtype=self.dtype)
                else:
                    x = paddle.static.data('input', (-1, self.in_channels, -1, -1, -1), dtype=self.dtype)
                y = paddle.static.nn.conv3d(x, self.out_channels, self.filter_shape, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, param_attr=paddle.nn.initializer.Assign(self.weight), bias_attr=False if self.no_bias else paddle.nn.initializer.Assign(self.bias), act=self.act, data_format=self.data_format)
        exe = base.Executor(self.place)
        exe.run(start)
        (out,) = exe.run(main, feed={'input': self.input}, fetch_list=[y])
        return out

    def static_graph_case_2(self):
        if False:
            i = 10
            return i + 15
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                if self.channel_last:
                    x = x = paddle.static.data('input', (-1, -1, -1, -1, self.in_channels), dtype=self.dtype)
                else:
                    x = paddle.static.data('input', (-1, self.in_channels, -1, -1, -1), dtype=self.dtype)
                weight = paddle.static.data('weight', self.weight.shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = paddle.static.data('bias', self.bias.shape, dtype=self.dtype)
                y = F.conv3d(x, weight, None if self.no_bias else bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
                if self.act == 'sigmoid':
                    y = F.sigmoid(y)
        exe = base.Executor(self.place)
        exe.run(start)
        feed_dict = {'input': self.input, 'weight': self.weight}
        if not self.no_bias:
            feed_dict['bias'] = self.bias
        (out,) = exe.run(main, feed=feed_dict, fetch_list=[y])
        return out

    def dygraph_case(self):
        if False:
            return 10
        with dg.guard(self.place):
            x = dg.to_variable(self.input)
            weight = dg.to_variable(self.weight)
            bias = None if self.no_bias else dg.to_variable(self.bias)
            y = F.conv3d(x, weight, bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
            if self.act == 'sigmoid':
                y = F.sigmoid(y)
            out = y.numpy()
        return out

    def _test_identity(self):
        if False:
            i = 10
            return i + 15
        self.prepare()
        out1 = self.static_graph_case_1()
        out2 = self.static_graph_case_2()
        out3 = self.dygraph_case()
        np.testing.assert_array_almost_equal(out1, out2)
        np.testing.assert_array_almost_equal(out2, out3)

    def test_identity_cpu(self):
        if False:
            print('Hello World!')
        self.place = base.CPUPlace()
        self._test_identity()

    @unittest.skipIf(not base.core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    def test_identity_gpu(self):
        if False:
            i = 10
            return i + 15
        self.place = base.CUDAPlace(0)
        self._test_identity()

class TestFunctionalConv3DError(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
    dtype = 'float32'

    def setUp(self):
        if False:
            return 10
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 'not_valid'
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

    def test_exception(self):
        if False:
            while True:
                i = 10
        self.prepare()
        with self.assertRaises(ValueError):
            self.static_graph_case()

    def prepare(self):
        if False:
            while True:
                i = 10
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape,) * 3
        else:
            filter_shape = tuple(self.filter_shape)
        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + filter_shape
        self.bias_shape = (self.out_channels,)

    def static_graph_case(self):
        if False:
            i = 10
            return i + 15
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                self.channel_last = self.data_format == 'NDHWC'
                if self.channel_last:
                    x = x = paddle.static.data('input', (-1, -1, -1, -1, self.in_channels), dtype=self.dtype)
                else:
                    x = paddle.static.data('input', (-1, self.in_channels, -1, -1, -1), dtype=self.dtype)
                weight = paddle.static.data('weight', self.weight_shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = paddle.static.data('bias', self.bias_shape, dtype=self.dtype)
                y = F.conv3d(x, weight, None if self.no_bias else bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
                if self.act == 'sigmoid':
                    y = F.sigmoid(y)

class TestFunctionalConv3DCase2(TestFunctionalConv3D):

    def setUp(self):
        if False:
            return 10
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

class TestFunctionalConv3DCase3(TestFunctionalConv3D):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 3, 1, 2, 3]
        self.stride = 2
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

class TestFunctionalConv3DCase4(TestFunctionalConv3D):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 1, 2, 2, 3, 3]
        self.stride = 1
        self.dilation = 2
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

class TestFunctionalConv3DCase5(TestFunctionalConv3D):

    def setUp(self):
        if False:
            return 10
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 1], [2, 2], [1, 1], [0, 0]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

class TestFunctionalConv3DCase6(TestFunctionalConv3D):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [1, 1], [2, 2], [2, 2]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DCase7(TestFunctionalConv3D):

    def setUp(self):
        if False:
            print('Hello World!')
        self.in_channels = 6
        self.out_channels = 8
        self.filter_shape = 3
        self.padding = 'same'
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DCase8(TestFunctionalConv3D):

    def setUp(self):
        if False:
            return 10
        self.in_channels = 6
        self.out_channels = 12
        self.filter_shape = 3
        self.padding = 'valid'
        self.stride = 1
        self.dilation = 1
        self.groups = 6
        self.no_bias = True
        self.act = None
        self.use_cudnn = False
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase2(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 1], [1, 2], [3, 4], [5, 6]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase3(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.in_channels = 3
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = 'same'
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'not_valid'

class TestFunctionalConv3DErrorCase4(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_channels = 4
        self.out_channels = 3
        self.filter_shape = 3
        self.padding = 'same'
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase7(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            print('Hello World!')
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 'same'
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'not_valid'

class TestFunctionalConv3DErrorCase8(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 1, 2, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase9(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.in_channels = -5
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [3, 2], [1, 2], [1, 1]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase10(TestFunctionalConv3DError):

    def setUp(self):
        if False:
            print('Hello World!')
        self.in_channels = 3
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = 'same'
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = 'sigmoid'
        self.data_format = 'NDHWC'

class TestFunctionalConv3DErrorCase11(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input = np.array([])
        self.filter = np.array([])
        self.num_filters = 0
        self.filter_size = 0
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = 'NCDHW'

    def static_graph_case(self):
        if False:
            print('Hello World!')
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                x = paddle.static.data('input', self.input.shape, dtype=paddle.float32)
                y = paddle.static.nn.conv3d(x, self.num_filters, self.filter_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, param_attr=paddle.nn.initializer.Assign(self.filter), bias_attr=False if self.bias is None else paddle.nn.initializer.Assign(self.bias), act=None, data_format=self.data_format)
        exe = base.Executor()
        exe.run(start)
        (out,) = exe.run(main, feed={'input': self.input}, fetch_list=[y])
        return out

    def dygraph_case(self):
        if False:
            return 10
        with dg.guard():
            x = dg.to_variable(self.input, dtype=paddle.float32)
            w = dg.to_variable(self.filter, dtype=paddle.float32)
            b = None if self.bias is None else dg.to_variable(self.bias, dtype=paddle.float32)
            y = F.conv3d(x, w, b, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)

    def test_dygraph_exception(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self.dygraph_case()

    def test_static_exception(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            self.static_graph_case()

class TestFunctionalConv3DErrorCase12(TestFunctionalConv3DErrorCase11):

    def setUp(self):
        if False:
            return 10
        self.input = np.random.randn(1, 3, 3, 3, 3)
        self.filter = np.random.randn(3, 3, 1, 1, 1)
        self.num_filters = 3
        self.filter_size = 1
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 0
        self.data_format = 'NCDHW'

class TestFunctionalConv3DErrorCase13(TestFunctionalConv3DErrorCase11):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input = np.random.randn(0, 0, 0, 0, 0)
        self.filter = np.random.randn(1, 0, 0, 0, 0)
        self.num_filters = 1
        self.filter_size = 1
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = 'NCDHW'
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()