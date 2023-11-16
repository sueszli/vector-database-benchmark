import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper
_inject_backend_tests = backend.inject_backend_tests(None, testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always']}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0']}))

@_inject_backend_tests
@testing.parameterize(*testing.product({'cover_all': [True, False], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contiguous': [None, 'C']}))
class TestMaxPooling2D(testing.FunctionTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if self.cover_all:
            self.output_shape = (2, 3, 3, 2)
        else:
            self.output_shape = (2, 3, 2, 2)
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 0.001, 'rtol': 0.01}
            self.check_double_backward_options = {'atol': 0.001, 'rtol': 0.01}
        else:
            self.check_backward_options = {'atol': 0.0001, 'rtol': 0.001}
            self.check_double_backward_options = {'atol': 0.0001, 'rtol': 0.001}

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        return (pooling_nd_helper.shuffled_linspace((2, 3, 4, 3), self.dtype),)

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        expect = numpy.empty(self.output_shape, dtype=self.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                if self.cover_all:
                    expect[i, c] = numpy.array([[xx[0:2, 0:2].max(), xx[0:2, 1:3].max()], [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()], [xx[3:4, 0:2].max(), xx[3:4, 1:3].max()]])
                else:
                    expect[i, c] = numpy.array([[xx[0:2, 0:2].max(), xx[0:2, 1:3].max()], [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()]])
        return (expect,)

    def forward(self, inputs, device):
        if False:
            while True:
                i = 10
        (x,) = inputs
        y = functions.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=self.cover_all)
        return (y,)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestMaxPooling2DForwardCpuWide(unittest.TestCase):

    def test_forward_cpu_wide(self):
        if False:
            print('Hello World!')
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = cuda.cupy.arange(2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        if False:
            while True:
                i = 10
        x = chainer.Variable(self.x)
        return functions.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=False)

    def test_call_cudnn_forward(self):
        if False:
            print('Hello World!')
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
                self.forward()
                self.assertEqual(func.called, chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backward(self):
        if False:
            for i in range(10):
                print('nop')
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)

class TestMaxPooling2DIndices(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = pooling_nd_helper.shuffled_linspace((2, 3, 4, 4), numpy.float32)

    def _check(self, x):
        if False:
            print('Hello World!')
        (out, indices) = functions.max_pooling_2d(x, 2, cover_all=False, return_indices=True)
        assert isinstance(out, chainer.Variable)
        assert isinstance(out.array, type(x))
        assert isinstance(indices, type(x))
        assert indices.shape == out.array.shape
        expect = numpy.zeros(indices.shape, dtype=indices.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                expect[i, c] = numpy.array([[xx[0:2, 0:2].ravel().argmax(), xx[0:2, 2:4].ravel().argmax()], [xx[2:4, 0:2].ravel().argmax(), xx[2:4, 2:4].ravel().argmax()]])
        if out.xp is cuda.cupy:
            expect = cuda.to_gpu(expect)
        assert (expect == indices).all()

    def test_cpu(self):
        if False:
            while True:
                i = 10
        self._check(self.x)

    @attr.gpu
    @attr.cudnn
    def test_gpu(self):
        if False:
            i = 10
            return i + 15
        x = cuda.to_gpu(self.x)
        with chainer.using_config('use_cudnn', 'never'):
            self._check(x)
        with chainer.using_config('use_cudnn', 'always'):
            self._check(x)
testing.run_module(__name__, __file__)