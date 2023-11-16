import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contiguous': [None, 'C']}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestReLU(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x,) = inputs
        return (functions.relu(x),)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        expected = x.copy()
        expected[expected < 0] = 0
        return (expected,)

@testing.parameterize(*testing.product({'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestReLUCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto')

    def forward(self):
        if False:
            for i in range(10):
                print('nop')
        x = chainer.Variable(self.x)
        return functions.relu(x)

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