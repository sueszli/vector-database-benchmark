import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
import paddle
from paddle import base
from paddle.base import core

class TestInstanceNormDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = 'float32'
            eps = 0.005
            atol = 0.0001
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.static.nn.instance_norm(input=x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check([x], z, x_init=x_arr, atol=atol, place=place, eps=eps)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestInstanceNormDoubleGradCheckWithoutParamBias(TestInstanceNormDoubleGradCheck):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = 'float32'
            eps = 0.005
            atol = 0.0001
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.static.nn.instance_norm(input=x, param_attr=False, bias_attr=False)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check([x], z, x_init=x_arr, atol=atol, place=place, eps=eps)

class TestInstanceNormDoubleGradEagerCheck(unittest.TestCase):

    def instance_norm_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.nn.functional.instance_norm(x[0])

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = 'float32'
            eps = 0.005
            atol = 0.0001
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.functional.instance_norm(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check([x], z, x_init=x_arr, atol=atol, place=place, eps=eps)
            gradient_checker.double_grad_check_for_dygraph(self.instance_norm_wrapper, [x], z, x_init=x_arr, atol=atol, place=place)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestInstanceNormDoubleGradEagerCheckWithParams(TestInstanceNormDoubleGradEagerCheck):

    def instance_norm_wrapper(self, x):
        if False:
            i = 10
            return i + 15
        instance_norm = paddle.nn.InstanceNorm2D(3)
        return instance_norm(x[0])

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = 'float32'
            eps = 0.005
            atol = 0.0001
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.InstanceNorm2D(3)(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check([x], z, x_init=x_arr, atol=atol, place=place, eps=eps)
            gradient_checker.double_grad_check_for_dygraph(self.instance_norm_wrapper, [x], z, x_init=x_arr, atol=atol, place=place)

class TestBatchNormDoubleGradCheck(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_test()

    def init_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.data_layout = 'NCHW'
        self.use_global_stats = False
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1

    def batch_norm_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        batch_norm = paddle.nn.BatchNorm2D(self.shape[self.channel_index], data_format=self.data_layout, use_global_stats=self.use_global_stats)
        return batch_norm(x[0])

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            dtype = 'float32'
            eps = 0.005
            atol = 0.0001
            x = paddle.create_parameter(dtype=dtype, shape=self.shape, name='x')
            z = paddle.static.nn.batch_norm(input=x, data_layout=self.data_layout, use_global_stats=self.use_global_stats)
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            gradient_checker.double_grad_check([x], z, x_init=x_arr, atol=atol, place=place, eps=eps)
            gradient_checker.double_grad_check_for_dygraph(self.batch_norm_wrapper, [x], z, x_init=x_arr, atol=atol, place=place)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestBatchNormDoubleGradCheckCase1(TestBatchNormDoubleGradCheck):

    def init_test(self):
        if False:
            return 10
        self.data_layout = 'NHWC'
        self.use_global_stats = False
        self.shape = [2, 3, 4, 5]
        self.channel_index = 3

class TestBatchNormDoubleGradCheckCase2(TestBatchNormDoubleGradCheck):

    def init_test(self):
        if False:
            return 10
        self.data_layout = 'NCHW'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1

class TestBatchNormDoubleGradCheckCase3(TestBatchNormDoubleGradCheck):

    def init_test(self):
        if False:
            print('Hello World!')
        self.data_layout = 'NHWC'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 3

class TestBatchNormDoubleGradCheckCase4(TestBatchNormDoubleGradCheck):

    def init_test(self):
        if False:
            print('Hello World!')
        self.data_layout = 'NCHW'
        self.use_global_stats = False
        self.shape = [2, 2, 3, 4, 5]
        self.channel_index = 1

    def batch_norm_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        batch_norm = paddle.nn.BatchNorm3D(self.shape[self.channel_index], data_format=self.data_layout, use_global_stats=self.use_global_stats)
        return batch_norm(x[0])

class TestBatchNormDoubleGradCheckCase5(TestBatchNormDoubleGradCheck):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed(37)
            dtype = 'float32'
            eps = 0.005
            atol = 0.0002
            chn = self.shape[1] if self.data_layout == 'NCHW' else self.shape[-1]
            x = paddle.create_parameter(dtype=dtype, shape=self.shape, name='x')
            z = paddle.static.nn.batch_norm(input=x, data_layout=self.data_layout, use_global_stats=self.use_global_stats)
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            (w, b) = prog.global_block().all_parameters()[1:3]
            w_arr = np.ones(chn).astype(dtype)
            b_arr = np.zeros(chn).astype(dtype)
            gradient_checker.double_grad_check([x, w, b], z, x_init=[x_arr, w_arr, b_arr], atol=atol, place=place, eps=eps)

class TestBatchNormDoubleGradCheckCase6(TestBatchNormDoubleGradCheckCase5):

    def init_test(self):
        if False:
            while True:
                i = 10
        self.data_layout = 'NCHW'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1
if __name__ == '__main__':
    unittest.main()