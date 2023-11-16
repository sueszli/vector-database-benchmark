import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
import paddle
from paddle import base
from paddle.base import core

class TestConvTransposeDoubleGradCheck(unittest.TestCase):

    def conv_transpose_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1)

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(x, 2, filter_size=1, groups=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps, atol=0.0001)
        else:
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv_transpose_wrapper, [x] + w, y, x_init=[x_arr] + w_arr, place=place)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestConvTranspose2DoubleGradCheck_AsyPadding(TestConvTransposeDoubleGradCheck):

    def conv_transpose_wrapper(self, x):
        if False:
            return 10
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1, padding=[1, 0, 0, 1])

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(input=x, num_filters=2, filter_size=1, padding=[1, 0, 0, 1], bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps, atol=0.0001)
        else:
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv_transpose_wrapper, [x] + w, y, x_init=[x_arr] + w_arr, place=place)

class TestConvTranspose2DoubleGradCheck_PaddingSAME(TestConvTransposeDoubleGradCheck):

    def conv_transpose_wrapper(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1, padding='SAME')

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(input=x, num_filters=2, filter_size=1, padding='SAME', bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps, atol=0.0001)
        else:
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv_transpose_wrapper, [x] + w, y, x_init=[x_arr] + w_arr, place=place)

class TestConvTranspose2DoubleGradCheck_PaddingVALID(TestConvTransposeDoubleGradCheck):

    def conv_transpose_wrapper(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1, padding='VALID')

    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(input=x, num_filters=2, filter_size=1, padding='VALID', bias_attr=False, use_cudnn=True)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps, atol=0.0001)
        else:
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv_transpose_wrapper, [x] + w, y, x_init=[x_arr] + w_arr, place=place)

class TestConvTranspose2DoubleGradCheck_ChannelLast(TestConvTransposeDoubleGradCheck):

    def conv_transpose_wrapper(self, x):
        if False:
            while True:
                i = 10
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1, padding=[1, 1], data_format='NHWC')

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        shape = [2, 3, 3, 2]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(input=x, num_filters=2, filter_size=1, padding=[1, 1], bias_attr=False, use_cudnn=True, groups=1, data_format='NHWC')
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps, atol=0.0001)
        else:
            gradient_checker.double_grad_check([x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.conv_transpose_wrapper, [x] + w, y, x_init=[x_arr] + w_arr, place=place)
if __name__ == '__main__':
    unittest.main()