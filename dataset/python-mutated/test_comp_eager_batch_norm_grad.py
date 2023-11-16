import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.base import core
np.random.seed(2023)

class Arg:
    dout = None

def generate_data(shape, dtype='float32'):
    if False:
        print('Hello World!')
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32'
        self.shape = [8, 8, 16, 16]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = 'NCHW'
        self.use_global_stats = None

    def set_dtype(self, dtype) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dtype = dtype

    def set_shape(self, shape) -> None:
        if False:
            while True:
                i = 10
        self.shape = shape

    def set_training(self, training) -> None:
        if False:
            print('Hello World!')
        self.training = training

    def set_momentum(self, momentum) -> None:
        if False:
            i = 10
            return i + 15
        self.momentum = momentum

    def set_epsilon(self, epsilon) -> None:
        if False:
            return 10
        self.epsilon = epsilon

    def set_data_format(self, data_format) -> None:
        if False:
            print('Hello World!')
        self.data_format = data_format

    def set_use_global_stats(self, use_global_stats) -> None:
        if False:
            print('Hello World!')
        self.use_global_stats = use_global_stats
attrs = Attr()

def fn(x, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats):
    if False:
        print('Hello World!')
    z = F.batch_norm(x, running_mean, running_variance, weight, bias, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format, use_global_stats=use_global_stats)
    out = z * paddle.to_tensor(Arg.dout)
    res = paddle.mean(out)
    return res

def expect_grad(x, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats):
    if False:
        for i in range(10):
            print('nop')
    x.stop_gradient = False
    res = fn(x, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats)
    gradients = paddle.grad(res, x)
    return gradients

def cal_composite(inputs, running_mean, running_variance, weight, bias):
    if False:
        print('Hello World!')
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x1 = paddle.static.data('x1', shape=inputs.shape, dtype=str(inputs.dtype))
        x1.stop_gradient = False
        x2 = paddle.static.data('x2', shape=running_mean.shape, dtype=str(running_mean.dtype))
        x3 = paddle.static.data('x3', shape=running_variance.shape, dtype=str(running_variance.dtype))
        x4 = paddle.static.data('x4', shape=weight.shape, dtype=str(weight.dtype))
        x5 = paddle.static.data('x5', shape=bias.shape, dtype=str(bias.dtype))
        y = fn(x1, x2, x3, x4, x5, attrs.training, attrs.momentum, attrs.epsilon, attrs.data_format, attrs.use_global_stats)
        z = paddle.static.gradients([y], [x1])
    exe = paddle.static.Executor()
    exe.run(startup_program)
    res = exe.run(main_program, feed={'x1': inputs, 'x2': running_mean, 'x3': running_variance, 'x4': weight, 'x5': bias}, fetch_list=[z])
    paddle.disable_static()
    return res

class TestCompositeBatchNorm(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtypes = ['float32']
        self.training = [False, True]
        self.shapes = [[8, 8, 16, 16], [2, 1, 2, 3]]
        self.momentum = [0.1, 0.9]
        self.epsilon = [1e-05, 2e-05]
        self.data_formats = ['NCHW']
        self.use_global_stats = [None, True, False]

    def compare_backward(self):
        if False:
            while True:
                i = 10
        if attrs.training is True and attrs.use_global_stats is False:
            return
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        Arg.dout = np.random.random(np_data.shape).astype(attrs.dtype)
        C = np_data.shape[1]
        running_mean = paddle.zeros(C, dtype=attrs.dtype)
        running_variance = paddle.ones(C, dtype=attrs.dtype)
        weight = paddle.ones(C, dtype=attrs.dtype) * 2
        bias = paddle.ones(C, dtype=attrs.dtype)
        expect = expect_grad(tensor_data, running_mean, running_variance, weight, bias, attrs.training, attrs.momentum, attrs.epsilon, attrs.data_format, attrs.use_global_stats)[0].numpy()
        np_running_mean = np.zeros(C, dtype=attrs.dtype)
        np_running_variance = np.ones(C, dtype=attrs.dtype)
        np_weight = np.ones(C, dtype=attrs.dtype) * 2
        np_bias = np.ones(C, dtype=attrs.dtype)
        actual = cal_composite(np_data, np_running_mean, np_running_variance, np_weight, np_bias)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=1e-05, atol=1e-05)

    def test_backward_prim_dygraph_vjp(self):
        if False:
            while True:
                i = 10
        core.set_prim_eager_enabled(True)
        for i in self.training:
            for j in self.dtypes:
                for m in self.momentum:
                    attrs.set_training(i)
                    attrs.set_dtype(j)
                    attrs.set_momentum(m)
                    self.compare_backward()
        for n in self.shapes:
            for t in self.use_global_stats:
                attrs.set_shape(n)
                attrs.set_use_global_stats(t)
                self.compare_backward()
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()