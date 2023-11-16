import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer import Variable

def _identiy_grid(in_shape):
    if False:
        i = 10
        return i + 15
    mesh = numpy.meshgrid(numpy.linspace(-1.0, 1.0, num=in_shape[2]), numpy.linspace(-1.0, 1.0, num=in_shape[3]))
    grid = numpy.concatenate([mesh[0][None], mesh[1][None]], axis=0)
    grid = numpy.repeat(grid[None], in_shape[0], axis=0).astype(numpy.float32)
    return grid

def _rotate_grid(in_shape):
    if False:
        print('Hello World!')
    mesh = numpy.meshgrid(numpy.linspace(-1.0, 1.0, num=in_shape[2]), numpy.linspace(-1.0, 1.0, num=in_shape[3]))
    mesh = [numpy.rot90(mesh[0]), numpy.rot90(mesh[1])]
    grid = numpy.concatenate([mesh[0][None], mesh[1][None]], axis=0)
    grid = numpy.repeat(grid[None], in_shape[0], axis=0).astype(numpy.float32)
    return grid

def _rotate_BCHW(x):
    if False:
        print('Hello World!')
    rotated_xs = []
    for i in range(x.shape[0]):
        x_i = x[i].transpose(1, 2, 0)
        x_i = numpy.rot90(x_i)
        rotated_xs.append(x_i.transpose(2, 0, 1))
    rotated_xs = numpy.concatenate([r_x[None] for r_x in rotated_xs], axis=0)
    return rotated_xs

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'use_cudnn': ['always', 'never']}))
class TestSpatialTransformerSampler(unittest.TestCase):
    in_shape = (2, 2, 4, 4)
    out_shape = (2, 2, 3, 3)
    grid_shape = (2, 2, 3, 3)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(size=self.in_shape).astype(self.dtype)
        self.grid = numpy.random.uniform(low=-2.0, high=2.0, size=self.grid_shape).astype(self.dtype)
        self.grads = numpy.random.uniform(size=self.out_shape).astype(self.dtype)

    def check_forward(self, x, grid):
        if False:
            while True:
                i = 10
        y = functions.spatial_transformer_sampler(x, grid)
        self.assertEqual(y.shape, self.out_shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_forward(self.x, self.grid)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid))

    def check_backward(self, x, grid, grads):
        if False:
            for i in range(10):
                print('nop')
        gradient_check.check_backward(functions.spatial_transformer_sampler, (x, grid), (grads,), dtype='d', atol=0.01, rtol=0.01, eps=1e-05)

    @condition.retry(3)
    def test_backward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_backward(self.x, self.grid, self.grads)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        if False:
            while True:
                i = 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid), cuda.to_gpu(self.grads))

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestSpatialTransformerSamplerConsistencyWithCuDNN(unittest.TestCase):
    in_shape = (2, 2, 4, 4)
    out_shape = (2, 2, 3, 3)
    grid_shape = (2, 2, 3, 3)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dtype == numpy.float16:
            uniform = numpy.random.RandomState(0).uniform
        else:
            uniform = numpy.random.uniform
        self.x = uniform(size=self.in_shape).astype(self.dtype)
        self.grid = uniform(low=-2, high=2, size=self.grid_shape).astype(self.dtype)
        self.grads = uniform(size=self.out_shape).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.assert_options = {'atol': 0.01}
        else:
            self.assert_options = {}

    def _apply_backward(self, x, grid, grads):
        if False:
            print('Hello World!')
        x = Variable(x)
        grid = Variable(grid)
        y = functions.spatial_transformer_sampler(x, grid)
        x.cleargrad()
        grid.cleargrad()
        y.grad = grads
        y.backward()
        return (x, grid, y)

    @attr.gpu
    @attr.cudnn
    def test_consistency_with_cudnn_cpu(self):
        if False:
            i = 10
            return i + 15
        with chainer.using_config('use_cudnn', 'never'):
            (x_cpu, grid_cpu, y_cpu) = self._apply_backward(self.x, self.grid, self.grads)
        with chainer.using_config('use_cudnn', 'always'):
            (x_cudnn, grid_cudnn, y_cudnn) = self._apply_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid), cuda.to_gpu(self.grads))
        testing.assert_allclose(y_cpu.data, y_cudnn.data, **self.assert_options)
        testing.assert_allclose(x_cpu.grad, x_cudnn.grad, **self.assert_options)
        testing.assert_allclose(grid_cpu.grad, grid_cudnn.grad, **self.assert_options)

    @attr.gpu
    @attr.cudnn
    def test_consistency_with_cudnn_gpu(self):
        if False:
            while True:
                i = 10
        with chainer.using_config('use_cudnn', 'never'):
            (x_gpu, grid_gpu, y_gpu) = self._apply_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid), cuda.to_gpu(self.grads))
        with chainer.using_config('use_cudnn', 'always'):
            (x_cudnn, grid_cudnn, y_cudnn) = self._apply_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid), cuda.to_gpu(self.grads))
        testing.assert_allclose(y_gpu.data, y_cudnn.data, **self.assert_options)
        testing.assert_allclose(x_gpu.grad, x_cudnn.grad, **self.assert_options)
        testing.assert_allclose(grid_gpu.grad, grid_cudnn.grad, **self.assert_options)

@testing.parameterize({'grid_creator': _identiy_grid, 'operator': lambda x: x, 'use_cudnn': 'always'}, {'grid_creator': _identiy_grid, 'operator': lambda x: x, 'use_cudnn': 'never'}, {'grid_creator': _rotate_grid, 'operator': _rotate_BCHW, 'use_cudnn': 'always'}, {'grid_creator': _rotate_grid, 'operator': _rotate_BCHW, 'use_cudnn': 'never'})
class TestSpatialTransformerSamplerForwardToyCases(unittest.TestCase):
    in_shape = (2, 2, 4, 4)
    grid_shape = (2, 2, 3, 3)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(size=self.in_shape).astype(numpy.float32)
        self.grid = self.grid_creator(self.in_shape)

    def check_forward(self, x, grid):
        if False:
            while True:
                i = 10
        y = functions.spatial_transformer_sampler(x, grid)
        testing.assert_allclose(y.data, self.operator(self.x))

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.x, self.grid)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid))

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'never']}))
class TestSpatialTransformerSamplerForwardPaddedImage(unittest.TestCase):
    in_shape = (1, 2, 4, 4)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(size=self.in_shape).astype(numpy.float32)
        p1 = [[-0.5], [-0.5]]
        p2 = [[3.5], [3.5]]
        p3 = [[2], [3.5]]
        p4 = [[-0.5], [2]]
        self.grid = numpy.concatenate((p1, p2, p3, p4), axis=1)
        self.grid = self.grid.reshape(1, 2, 4, 1).astype(numpy.float32)
        self.grid[:, 0] = (self.grid[:, 0] / (self.in_shape[3] - 1) - 0.5) * 2
        self.grid[:, 1] = (self.grid[:, 1] / (self.in_shape[2] - 1) - 0.5) * 2
        exp_p1 = self.x[0, :, 0, 0] / 4
        exp_p2 = self.x[0, :, 3, 3] / 4
        exp_p3 = self.x[0, :, 3, 2] / 2
        exp_p4 = self.x[0, :, 2, 0] / 2
        self.expected = numpy.concatenate((exp_p1[:, None], exp_p2[:, None], exp_p3[:, None], exp_p4[:, None]), axis=1)
        self.expected = self.expected.reshape(1, 2, 4, 1).astype(numpy.float32)

    def check_forward(self, x, grid, expected):
        if False:
            i = 10
            return i + 15
        y = functions.spatial_transformer_sampler(x, grid)
        testing.assert_allclose(y.data, expected)

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            print('Hello World!')
        self.check_forward(self.x, self.grid, self.expected)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            while True:
                i = 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid), cuda.to_gpu(self.expected))
testing.run_module(__name__, __file__)