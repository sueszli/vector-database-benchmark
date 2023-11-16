import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn

def _reverse_repeat_list(t, n):
    if False:
        print('Hello World!')
    return [x for x in reversed(t) for _ in range(n)]

class Conv2DTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', batch_size=4, spartial_shape=(16, 16), num_channels=6, num_filters=8, filter_size=3, padding=0, padding_mode='zeros', stride=1, dilation=1, groups=1, no_bias=False, data_format='NCHW', dtype='float32'):
        if False:
            i = 10
            return i + 15
        super().__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spartial_shape = spartial_shape
        self.filter_size = filter_size
        self.padding = padding
        if padding_mode in {'reflect', 'replicate', 'circular'}:
            _paired_padding = paddle.utils.convert_to_list(padding, 2, 'padding')
            self._reversed_padding_repeated_twice = _reverse_repeat_list(_paired_padding, 2)
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.data_format = data_format
        self.dtype = dtype

    def setUp(self):
        if False:
            return 10
        self.channel_last = self.data_format == 'NHWC'
        if self.channel_last:
            input_shape = (self.batch_size,) + self.spartial_shape + (self.num_channels,)
        else:
            input_shape = (self.batch_size, self.num_channels) + self.spartial_shape
        self.input = np.random.randn(*input_shape).astype(self.dtype)
        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size] * 2
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (self.num_filters, self.num_channels // self.groups) + tuple(filter_size)
        self.weight = np.random.uniform(-1, 1, size=weight_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, size=(self.num_filters,)).astype(self.dtype)
        else:
            self.bias = None

    def base_layer(self, place):
        if False:
            print('Hello World!')
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                input_shape = (-1, -1, -1, self.num_channels) if self.channel_last else (-1, self.num_channels, -1, -1)
                x_var = paddle.static.data('input', input_shape, dtype=self.dtype)
                weight_attr = paddle.nn.initializer.Assign(self.weight)
                if self.bias is None:
                    bias_attr = False
                else:
                    bias_attr = paddle.nn.initializer.Assign(self.bias)
                if self.padding_mode != 'zeros':
                    x_var = F.pad(x_var, self._reversed_padding_repeated_twice, mode=self.padding_mode, data_format=self.data_format)
                    padding = 0
                else:
                    padding = self.padding
                y_var = paddle.static.nn.conv2d(x_var, self.num_filters, self.filter_size, padding=padding, stride=self.stride, dilation=self.dilation, groups=self.groups, param_attr=weight_attr, bias_attr=bias_attr, data_format=self.data_format)
        feed_dict = {'input': self.input}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        if False:
            while True:
                i = 10
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                input_shape = (-1, -1, -1, self.num_channels) if self.channel_last else (-1, self.num_channels, -1, -1)
                x_var = paddle.static.data('input', input_shape, dtype=self.dtype)
                w_var = paddle.static.data('weight', self.weight_shape, dtype=self.dtype)
                b_var = paddle.static.data('bias', (self.num_filters,), dtype=self.dtype)
                if self.padding_mode != 'zeros':
                    x_var = F.pad(x_var, self._reversed_padding_repeated_twice, mode=self.padding_mode, data_format=self.data_format)
                    padding = 0
                else:
                    padding = self.padding
                y_var = F.conv2d(x_var, w_var, b_var if not self.no_bias else None, padding=padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
        feed_dict = {'input': self.input, 'weight': self.weight}
        if self.bias is not None:
            feed_dict['bias'] = self.bias
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_nn_layer(self):
        if False:
            for i in range(10):
                print('nop')
        x_var = paddle.to_tensor(self.input)
        x_var.stop_gradient = False
        conv = nn.Conv2D(self.num_channels, self.num_filters, self.filter_size, padding=self.padding, padding_mode=self.padding_mode, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var)
        y_var.backward()
        y_np = y_var.numpy()
        t1 = x_var.gradient()
        return (y_np, t1)

    def run_Conv2D_static(self, place):
        if False:
            while True:
                i = 10
        paddle.seed(2023)
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                x_var = paddle.static.data('input', self.input.shape, dtype=self.dtype)
                conv = nn.Conv2D(self.num_channels, self.num_filters, self.filter_size, padding=self.padding, padding_mode=self.padding_mode, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
                y_var = conv(x_var)
        feed_dict = {'input': self.input}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def run_Conv2D_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2023)
        x_var = paddle.to_tensor(self.input)
        x_var.stop_gradient = False
        conv = nn.Conv2D(self.num_channels, self.num_filters, self.filter_size, padding=self.padding, padding_mode=self.padding_mode, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)
        y_var = conv(x_var)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        if False:
            for i in range(10):
                print('nop')
        result1 = self.base_layer(place)
        result2 = self.functional(place)
        with dg.guard(place):
            (result3, g1) = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def _test_equivalence_in_pir(self, place):
        if False:
            return 10
        with paddle.pir_utils.IrGuard():
            result1 = self.run_Conv2D_static(place)
            with dg.guard(place):
                result2 = self.run_Conv2D_dygraph()
            np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        if False:
            return 10
        place = base.CPUPlace()
        self._test_equivalence(place)
        self._test_equivalence_in_pir(place)
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)
            self._test_equivalence_in_pir(place)

class Conv2DErrorTestCase(Conv2DTestCase):

    def runTest(self):
        if False:
            return 10
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()

def add_cases(suite):
    if False:
        i = 10
        return i + 15
    suite.addTest(Conv2DTestCase(methodName='runTest'))
    suite.addTest(Conv2DTestCase(methodName='runTest', stride=[1, 2], dilation=2))
    suite.addTest(Conv2DTestCase(methodName='runTest', stride=2, dilation=(2, 1)))
    suite.addTest(Conv2DTestCase(methodName='runTest', padding='same', no_bias=True))
    suite.addTest(Conv2DTestCase(methodName='runTest', filter_size=(3, 3), padding='valid'))
    suite.addTest(Conv2DTestCase(methodName='runTest', padding=(2, 3)))
    suite.addTest(Conv2DTestCase(methodName='runTest', padding=[1, 2, 2, 1]))
    suite.addTest(Conv2DTestCase(methodName='runTest', padding=[[0, 0], [0, 0], [1, 2], [2, 1]]))
    suite.addTest(Conv2DTestCase(methodName='runTest', data_format='NHWC'))
    suite.addTest(Conv2DTestCase(methodName='runTest', data_format='NHWC', padding=[[0, 0], [1, 1], [2, 2], [0, 0]]))
    suite.addTest(Conv2DTestCase(methodName='runTest', groups=2, padding='valid'))
    suite.addTest(Conv2DTestCase(methodName='runTest', num_filters=6, num_channels=3, groups=3, padding='valid'))
    suite.addTest(Conv2DTestCase(methodName='runTest', filter_size=(3, 3), padding=1, padding_mode='reflect'))
    suite.addTest(Conv2DTestCase(methodName='runTest', filter_size=(3, 3), padding=1, padding_mode='replicate'))
    suite.addTest(Conv2DTestCase(methodName='runTest', filter_size=(3, 3), padding=1, padding_mode='circular'))

def add_error_cases(suite):
    if False:
        for i in range(10):
            print('nop')
    suite.addTest(Conv2DErrorTestCase(methodName='runTest', num_channels=5, groups=2))
    suite.addTest(Conv2DErrorTestCase(methodName='runTest', num_channels=5, groups=2, stride=0))
    suite.addTest(Conv2DErrorTestCase(methodName='runTest', num_channels=5, groups=2, padding=[-1, -1]))

def load_tests(loader, standard_tests, pattern):
    if False:
        return 10
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()