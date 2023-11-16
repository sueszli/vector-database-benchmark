import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper

@testing.parameterize(*testing.product_dict([{'pyramid_height': 3, 'output_dim': 63, 'n': 2, 'c': 3, 'h': 9, 'w': 8}], [{'pooling': 'max'}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
class TestSpatialPyramidPooling2D(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        shape = (self.n, self.c, self.h, self.w)
        self.x = pooling_nd_helper.shuffled_linspace(shape, self.dtype)
        self.one = numpy.ones((self.n, self.c, self.h, self.w)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (self.n, self.output_dim, 1, 1)).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

    def func(self, x):
        if False:
            return 10
        return functions.spatial_pyramid_pooling_2d(x, self.pyramid_height, pooling=self.pooling)

    def check_forward(self, x_data, use_cudnn='always'):
        if False:
            return 10
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = self.func(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)
        self.assertEqual(self.gy.shape, y_data.shape)

    def check_forward_ones(self, x_data, use_cudnn='always'):
        if False:
            while True:
                i = 10
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = self.func(x)
        y_data = cuda.to_cpu(y.data)
        self.assertEqual(y_data.shape, (self.n, self.output_dim, 1, 1))
        self.assertEqual(y_data.dtype, self.dtype)
        testing.assert_allclose(y_data, numpy.ones_like(y_data))

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.x)
        self.check_forward_ones(self.one)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            print('Hello World!')
        self.check_forward(cuda.to_gpu(self.x))
        self.check_forward_ones(cuda.to_gpu(self.one))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        if False:
            print('Hello World!')
        self.check_forward(cuda.to_gpu(self.x), 'never')
        self.check_forward_ones(cuda.to_gpu(self.one), 'never')

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        if False:
            print('Hello World!')
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(self.func, x_data, y_grad, dtype=numpy.float64, atol=0.0005, rtol=0.005)

    @condition.retry(3)
    def test_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        if False:
            return 10
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        if False:
            print('Hello World!')
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    def check_double_backward(self, x_data, y_grad, x_grad_grad, use_cudnn='always'):
        if False:
            for i in range(10):
                print('nop')
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(self.func, x_data, y_grad, x_grad_grad, dtype=numpy.float64, atol=0.005, rtol=0.005)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_double_backward(self.x, self.gy, self.ggx, 'never')

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_non_contiguous(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)), cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)), cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_no_cudnn(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx), 'never')

class TestInvalidDtype(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = numpy.random.randn(5, 3, 5, 5)
        self.v = chainer.Variable(self.x.astype(numpy.int32))

    def check_invalid_dtype(self):
        if False:
            while True:
                i = 10
        functions.spatial_pyramid_pooling_2d(self.v, 3, pooling='max')

    def test_invalid_dtype_cpu(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()

    @attr.gpu
    def test_invalid_dtype_gpu(self):
        if False:
            print('Hello World!')
        self.v.to_gpu()
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()

class TestInvalidArguments(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = numpy.random.randn(5, 3, 5, 5)
        self.v = chainer.Variable(self.x.astype(numpy.float32))

    def check_ambiguous_poolings(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            functions.spatial_pyramid_pooling_2d(self.v, 3)

    def check_invalid_poolings(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            functions.spatial_pyramid_pooling_2d(self.v, 3, pooling='avg')

    def test_ambiguous_pooling(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_ambiguous_poolings()

    def test_invalid_pooling(self):
        if False:
            print('Hello World!')
        self.check_invalid_poolings()

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.gpu
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        shape = (2, 3, 9, 8)
        size = 2 * 3 * 9 * 8
        self.x = cuda.cupy.arange(size, dtype=self.dtype).reshape(shape)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 63, 1, 1)).astype(self.dtype)

    def forward(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(self.x)
        return functions.spatial_pyramid_pooling_2d(x, 3, pooling='max')

    def test_call_cudnn_forward(self):
        if False:
            for i in range(10):
                print('nop')
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
                self.forward()
                self.assertEqual(func.called, chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backward(self):
        if False:
            print('Hello World!')
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)
testing.run_module(__name__, __file__)