import functools
import itertools
import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

def _skip_if(cond, reason):
    if False:
        print('Hello World!')
    'Skip test if cond(self) is True'

    def decorator(impl):
        if False:
            while True:
                i = 10

        @functools.wraps(impl)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if cond(self):
                raise unittest.SkipTest(reason)
            else:
                impl(self, *args, **kwargs)
        return wrapper
    return decorator

def _is_good_param(param):
    if False:
        i = 10
        return i + 15
    return param['nonzeros'] is None or param['nonzeros'] < numpy.prod(param['shape'])

@testing.parameterize(*filter(_is_good_param, testing.product([[{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}], [{'shape': (4, 15), 'axis': 1}, {'shape': (4,), 'axis': 0}, {'shape': (4, 3, 2, 5), 'axis': 0}, {'shape': (4, 3, 2, 5), 'axis': 1}, {'shape': (4, 3, 2, 5), 'axis': 2}, {'shape': (4, 3, 2, 5), 'axis': 3}, {'shape': (4, 3, 2), 'axis': (0, 1)}, {'shape': (4, 3, 2, 4, 3, 2, 2), 'axis': (1, 4, 3, 6)}, {'shape': (0, 2), 'axis': 1}, {'shape': (), 'axis': ()}], [{'eps': 1e-05, 'nonzeros': None}, {'eps': 0.1, 'nonzeros': None}, {'eps': 0.1, 'nonzeros': 0, 'truezero': True}, {'eps': 0.1, 'nonzeros': 0, 'truezero': False}, {'eps': 0.1, 'nonzeros': 2, 'truezero': True}, {'eps': 0.1, 'nonzeros': 2, 'truezero': False}]])))
class TestL2Normalization(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        min_abs = 0.1
        if self.dtype == numpy.float16:
            tuple_axis = self.axis
            if not isinstance(tuple_axis, tuple):
                tuple_axis = (tuple_axis,)
            aggr_size = numpy.prod([self.shape[i] for i in tuple_axis], dtype=int)
            min_abs = max(min_abs, 0.5 / aggr_size)
        self.x = chainer.utils.force_array(numpy.random.uniform(min_abs, 1, self.shape) * (1 - 2 * numpy.random.randint(2, size=self.shape)), self.dtype)
        if self.nonzeros is not None:
            zeros = self.x.size - self.nonzeros
            while True:
                rand = numpy.random.uniform(0, 1, self.shape)
                mask = rand <= numpy.sort(rand.ravel())[zeros - 1]
                if self.x[mask].shape == (zeros,):
                    break
            if self.truezero:
                self.x[mask] = 0
            else:
                zero_scale = 10.0 ** numpy.random.randint(-40, -3)
                self.x[mask] = numpy.random.uniform(-zero_scale, zero_scale, zeros)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.001, 'rtol': 0.001}
        else:
            self.check_forward_options = {}
        if self.nonzeros is None:
            if self.dtype == numpy.float16:
                self.check_backward_options = {'dtype': numpy.float64, 'atol': 0.005, 'rtol': 0.005}
                self.check_double_backward_options = {'dtype': numpy.float64, 'atol': 0.01, 'rtol': 0.01}
            else:
                self.check_backward_options = {'dtype': numpy.float64, 'atol': 0.0001, 'rtol': 0.0001}
                self.check_double_backward_options = {'dtype': numpy.float64, 'atol': 0.0001, 'rtol': 0.0001}
        else:
            self.check_backward_options = {'dtype': numpy.float64, 'atol': 0.01, 'rtol': 0.01, 'eps': 0.0001}

    def check_forward(self, x_data, axis):
        if False:
            i = 10
            return i + 15
        eps = self.eps
        x = chainer.Variable(x_data)
        y = functions.normalize(x, eps=eps, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)
        y_expect = numpy.empty_like(self.x)
        shape = self.x.shape
        indices = []
        axis_tuple = axis if isinstance(axis, tuple) else (axis,)
        for i in six.moves.range(len(shape)):
            if i not in axis_tuple:
                indices.append(six.moves.range(shape[i]))
            else:
                indices.append([slice(None)])
        indices_tuple = list(itertools.product(*indices))
        for index in indices_tuple:
            numerator = numpy.linalg.norm(self.x[index]).astype(x.dtype) + eps
            y_expect[index] = self.x[index] / numerator
        testing.assert_allclose(y_expect, y_data, **self.check_forward_options)

    def test_forward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_forward(self.x, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            return 10
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return functions.normalize(x, eps=self.eps, axis=axis)
        gradient_check.check_backward(f, x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            return 10
        self.check_backward(cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    @_skip_if(lambda self: self.nonzeros is not None, 'backward of L2Normalize is non-differentiable at zero vector')
    def check_double_backward(self, x_data, axis, y_grad, x_grad_grad):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return functions.normalize(x, eps=self.eps, axis=axis)
        gradient_check.check_double_backward(f, x_data, y_grad, x_grad_grad, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        if False:
            return 10
        self.check_double_backward(self.x, self.axis, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        if False:
            return 10
        self.check_double_backward(cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def check_eps(self, x_data):
        if False:
            return 10
        x = chainer.Variable(x_data)
        y = functions.normalize(x, axis=self.axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)
        y_expect = numpy.zeros_like(self.x)
        testing.assert_allclose(y_expect, y_data)

    def test_eps_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_eps(numpy.zeros_like(self.x))

    @attr.gpu
    def test_eps_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_eps(cuda.to_gpu(numpy.zeros_like(self.x)))
testing.run_module(__name__, __file__)