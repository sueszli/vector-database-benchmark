import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contiguous': [None, 'C']}))
@backend.inject_backend_tests(None, testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always']}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}])
class TestAveragePooling2D(testing.FunctionTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.0005, 'rtol': 0.005}
            self.check_backward_options = {'atol': 0.0005, 'rtol': 0.005}
            self.check_double_backward_options = {'atol': 0.0005, 'rtol': 0.005}

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        return (functions.average_pooling_2d(x, 3, stride=2, pad=1),)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y = numpy.empty((2, 3, 2, 2), dtype=self.dtype)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[k, c]
                y[k, c] = numpy.array([[xx[0:2, 0:2].sum(), xx[0:2, 1:3].sum()], [xx[1:4, 0:2].sum(), xx[1:4, 1:3].sum()]]) / 9
        return (y,)

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestAveragePooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = cuda.cupy.arange(2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(self.x)
        return functions.average_pooling_2d(x, 3, stride=2, pad=1)

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
            return 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)
testing.run_module(__name__, __file__)