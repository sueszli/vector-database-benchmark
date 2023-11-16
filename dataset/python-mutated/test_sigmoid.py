import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer import utils

def _sigmoid(x):
    if False:
        print('Hello World!')
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contiguous': [None, 'C']}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}, {'use_ideep': True}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestSigmoid(testing.FunctionTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.0001, 'rtol': 0.001}
            self.check_backward_options = {'atol': 0.01, 'rtol': 0.05}
            self.check_double_backward_options = {'atol': 0.01, 'rtol': 0.05}

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x,) = inputs
        y = _sigmoid(x)
        y = utils.force_array(y)
        return (y,)

    def forward(self, inputs, device):
        if False:
            return 10
        (x,) = inputs
        return (functions.sigmoid(x),)

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestSigmoidCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('==always')

    def forward(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(self.x)
        return functions.sigmoid(x)

    def test_call_cudnn_forward(self):
        if False:
            while True:
                i = 10
        default_func = cuda.cupy.cudnn.activation_forward
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.activation_forward') as func:
                func.side_effect = default_func
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        if False:
            return 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            default_func = cuda.cupy.cudnn.activation_backward
            with testing.patch('cupy.cudnn.activation_backward') as func:
                func.side_effect = default_func
                y.backward()
                self.assertEqual(func.called, self.expect)
testing.run_module(__name__, __file__)