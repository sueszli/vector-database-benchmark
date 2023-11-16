import unittest
import numpy as np
import paddle
paddle.enable_static()
from paddle import base
from paddle.base import core

class TestConv2DAPI(unittest.TestCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        input_NHWC = paddle.static.data(name='input_NHWC', shape=[2, 5, 5, 3], dtype='float32')
        input_NCHW = paddle.static.data(name='input_NCHW', shape=[2, 3, 5, 5], dtype='float32')
        paddle.static.nn.conv2d(input=input_NHWC, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=1, data_format='NCHW')
        paddle.static.nn.conv2d(input=input_NCHW, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=[1, 2, 1, 0], dilation=[1, 1], groups=1, data_format='NCHW')
        paddle.static.nn.conv2d(input=input_NCHW, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=[[0, 0], [0, 0], [1, 1], [1, 1]], dilation=[1, 1], groups=1, data_format='NCHW')
        paddle.static.nn.conv2d(input=input_NHWC, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0]], dilation=[1, 1], groups=1, data_format='NHWC')
        paddle.static.nn.conv2d(input=input_NCHW, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding='SAME', dilation=[1, 1], groups=1, data_format='NCHW')
        paddle.static.nn.conv2d(input=input_NCHW, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding='VALID', dilation=[1, 1], groups=1, data_format='NCHW')

    def test_depthwise_conv2d(self):
        if False:
            for i in range(10):
                print('nop')
        x_var = paddle.uniform((2, 8, 8, 4), dtype='float32', min=-1.0, max=1.0)
        conv = paddle.nn.Conv2D(in_channels=4, out_channels=4, kernel_size=(3, 3), groups=4, data_format='NHWC')
        y_var = conv(x_var)

class TestConv2DAPI_Error(unittest.TestCase):

    def test_api(self):
        if False:
            return 10
        input = paddle.static.data(name='input', shape=[2, 5, 5, 5], dtype='float32')

        def run_1():
            if False:
                return 10
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=1, use_cudnn=[0], data_format='NCHW')
        self.assertRaises(ValueError, run_1)

        def run_2():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=1, use_cudnn=False, data_format='NCHWC')
        self.assertRaises(ValueError, run_2)

        def run_3():
            if False:
                i = 10
                return i + 15
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding='SAMEE', dilation=[1, 1], groups=1, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_3)

        def run_4():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=[[0, 1], [0, 1], [0, 1], [0, 1]], dilation=[1, 1], groups=1, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_4)

        def run_5():
            if False:
                return 10
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=[[0, 1], [0, 1], [0, 1], [0, 1]], dilation=[1, 1], groups=1, use_cudnn=False, data_format='NHWC')
        self.assertRaises(ValueError, run_5)
        x = paddle.static.data(name='x', shape=[2, 5, 5, -1], dtype='float32')

        def run_6():
            if False:
                for i in range(10):
                    print('nop')
            paddle.static.nn.conv2d(input=x, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=1, use_cudnn=False, data_format='NHWC')
        self.assertRaises(ValueError, run_6)

        def run_7():
            if False:
                for i in range(10):
                    print('nop')
            paddle.static.nn.conv2d(input=input, num_filters=3, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=3, use_cudnn=False, data_format='NHWC')
        self.assertRaises(ValueError, run_7)

        def run_8():
            if False:
                i = 10
                return i + 15
            paddle.static.nn.conv2d(input=input, num_filters=0, filter_size=0, stride=0, padding=0, dilation=0, groups=1, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_8)

        def run_9():
            if False:
                i = 10
                return i + 15
            paddle.static.nn.conv2d(input=input, num_filters=0, filter_size=0, stride=0, padding=0, dilation=0, groups=0, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_9)

        def run_10():
            if False:
                while True:
                    i = 10
            paddle.static.nn.conv2d(input=input, num_filters=1, filter_size=1, stride=0, padding=0, dilation=0, groups=1, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_10)

    def test_api_with_error_input(self):
        if False:
            print('Hello World!')
        input = paddle.static.data(name='error_input', shape=[1], dtype='float32')

        def run_1():
            if False:
                print('Hello World!')
            paddle.static.nn.conv2d(input=input, num_filters=0, filter_size=0, stride=0, padding=0, dilation=0, groups=0, use_cudnn=False, data_format='NCHW')
        self.assertRaises(ValueError, run_1)

@unittest.skipIf(not (core.is_compiled_with_cuda() or core.is_compiled_with_rocm()), 'core is not compiled with CUDA or ROCM')
class TestConv2DEnviron(unittest.TestCase):

    def run1(self, place):
        if False:
            i = 10
            return i + 15
        with base.program_guard(base.Program(), base.Program()):
            inputs = paddle.static.data(shape=[2, 3, 5, 5], name='inputs', dtype='float32')
            result = paddle.static.nn.conv2d(input=inputs, num_filters=4, filter_size=[3, 3], stride=[1, 1], padding=0, dilation=[1, 1], groups=1, data_format='NCHW')
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            fetches = exe.run(base.default_main_program(), feed={'inputs': self.input_np}, fetch_list=[result])

    def run2(self, place):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(place):
            inputs = base.dygraph.to_variable(self.input_np)
            conv = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=(3, 3), data_format='NCHW')
            result = conv(inputs)

    def run_all(self, place):
        if False:
            i = 10
            return i + 15
        self.run1(place)
        self.run2(place)

    def test_environ(self):
        if False:
            while True:
                i = 10
        self.input_np = np.random.random([2, 3, 5, 5]).astype('float32')
        for place in [paddle.CPUPlace(), paddle.CUDAPlace(0)]:
            base.set_flags({'FLAGS_conv2d_disable_cudnn': False})
            self.run_all(place)
            base.set_flags({'FLAGS_conv2d_disable_cudnn': True})
            self.run_all(place)
if __name__ == '__main__':
    unittest.main()