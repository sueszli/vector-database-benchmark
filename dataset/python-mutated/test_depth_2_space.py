import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestDepth2Space(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.depth = numpy.arange(96).reshape(2, 8, 3, 2).astype(self.dtype)
        self.space = numpy.array([[[[0.0, 12.0, 1.0, 13.0], [24.0, 36.0, 25.0, 37.0], [2.0, 14.0, 3.0, 15.0], [26.0, 38.0, 27.0, 39.0], [4.0, 16.0, 5.0, 17.0], [28.0, 40.0, 29.0, 41.0]], [[6.0, 18.0, 7.0, 19.0], [30.0, 42.0, 31.0, 43.0], [8.0, 20.0, 9.0, 21.0], [32.0, 44.0, 33.0, 45.0], [10.0, 22.0, 11.0, 23.0], [34.0, 46.0, 35.0, 47.0]]], [[[48.0, 60.0, 49.0, 61.0], [72.0, 84.0, 73.0, 85.0], [50.0, 62.0, 51.0, 63.0], [74.0, 86.0, 75.0, 87.0], [52.0, 64.0, 53.0, 65.0], [76.0, 88.0, 77.0, 89.0]], [[54.0, 66.0, 55.0, 67.0], [78.0, 90.0, 79.0, 91.0], [56.0, 68.0, 57.0, 69.0], [80.0, 92.0, 81.0, 93.0], [58.0, 70.0, 59.0, 71.0], [82.0, 94.0, 83.0, 95.0]]]]).astype(self.dtype)
        self.x = numpy.random.randn(2, 8, 3, 2).astype(self.dtype)
        self.gy = numpy.random.randn(2, 2, 6, 4).astype(self.dtype)
        self.ggx = numpy.random.randn(2, 8, 3, 2).astype(self.dtype)
        self.r = 2
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 0.0005, 'rtol': 0.005}
            self.check_double_backward_options = {'atol': 0.005, 'rtol': 0.05}

    def check_forward(self, depth_data, space_data):
        if False:
            for i in range(10):
                print('nop')
        depth = chainer.Variable(depth_data)
        d2s = functions.depth2space(depth, self.r)
        d2s_value = cuda.to_cpu(d2s.data)
        self.assertEqual(d2s_value.dtype, self.dtype)
        self.assertEqual(d2s_value.shape, (2, 2, 6, 4))
        d2s_expect = space_data
        testing.assert_allclose(d2s_value, d2s_expect)

    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.depth, self.space)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_forward(cuda.to_gpu(self.depth), cuda.to_gpu(self.space))

    def check_backward(self, x_data, y_grad):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return functions.depth2space(x, self.r)
        gradient_check.check_backward(f, x_data, y_grad, dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return functions.depth2space(x, self.r)
        gradient_check.check_double_backward(f, x_data, y_grad, x_grad_grad, dtype=numpy.float64, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))
testing.run_module(__name__, __file__)