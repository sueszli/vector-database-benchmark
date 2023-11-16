import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
import paddle
from paddle import base
from paddle.base import core

class TestConvDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(x, 2, 1, groups=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConvDoubleGradCheckTest0(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(x, 2, 1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConvDoubleGradCheckTest1(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        shape = [2, 3, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(x, 2, 1, padding=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        shape = [2, 4, 3, 4, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(x, 2, 1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheckTest1(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 4, 5, 3, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(x, 2, 1, padding=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv2DoubleGradCheck_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(input=x, num_filters=2, filter_size=1, padding=[1, 0, 0, 1], bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv2DoubleGradCheck_PaddingSAME(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(input=x, num_filters=2, filter_size=1, padding='SAME', bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            print('Hello World!')
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv2DoubleGradCheck_PaddingVALID(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(input=x, num_filters=2, filter_size=1, padding='VALID', bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv2DoubleGradCheck_ChannelLast(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(input=x, num_filters=2, filter_size=1, padding=[1, 1], bias_attr=False, use_cudnn=True, groups=1, data_format='NHWC')
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv2DoubleGradCheck_ChannelLast_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(input=x, num_filters=2, filter_size=1, padding=[1, 0, 1, 0], bias_attr=False, use_cudnn=True, groups=1, data_format='NHWC')
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheck_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 2, 2, 2, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(input=x, num_filters=2, filter_size=1, padding=[1, 0, 0, 1, 1, 2], bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DoubleGradCheck_PaddingSAME(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 2, 2, 2, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(input=x, num_filters=2, filter_size=1, padding='SAME', groups=1, bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DoubleGradCheck_PaddingVALID(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 2, 3, 3, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(input=x, num_filters=2, filter_size=1, padding='VALID', bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheck_ChannelLast(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 2, 2, 2, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(input=x, num_filters=2, filter_size=1, padding=[1, 1, 1], bias_attr=False, use_cudnn=True, groups=1, data_format='NDHWC')
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheck_ChannelLast_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        shape = [2, 2, 2, 2, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv3d(input=x, num_filters=2, filter_size=1, padding=[1, 0, 1, 0, 1, 0], bias_attr=False, use_cudnn=True, groups=1, data_format='NDHWC')
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestDepthWiseConvDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d(x, shape[1], 1, groups=shape[1], bias_attr=False, use_cudnn=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestDepthWiseConvDoubleGradCheckCase1(unittest.TestCase):

    def depthwise_conv2d_wrapper(self, x):
        if False:
            while True:
                i = 10
        return paddle.nn.functional.conv2d(x[0], x[1], groups=4)

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        x_shape = [2, 4, 3, 3]
        w_shape = [4, 1, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = paddle.nn.functional.conv2d(x, w, groups=4)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)
        gradient_checker.double_grad_check([x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.depthwise_conv2d_wrapper, [x, w], y, x_init=[x_arr, w_arr], place=place)

    def test_grad(self):
        if False:
            return 10
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConv3DDoubleGradCheck_NN(unittest.TestCase):

    def conv3d_wrapper(self, x):
        if False:
            print('Hello World!')
        return paddle.nn.functional.conv3d(x[0], x[1])

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        x_shape = [2, 3, 8, 8, 8]
        w_shape = [6, 3, 3, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = paddle.nn.functional.conv3d(x, w)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)
        gradient_checker.double_grad_check([x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv3d_wrapper, [x, w], y, x_init=[x_arr, w_arr], place=place)

    def test_grad(self):
        if False:
            while True:
                i = 10
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()