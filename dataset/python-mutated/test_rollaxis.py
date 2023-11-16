import unittest
import numpy
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check

@testing.parameterize({'axis': 0, 'start': 2, 'out_shape': (3, 2, 4)}, {'axis': 2, 'start': 0, 'out_shape': (4, 2, 3)}, {'axis': 1, 'start': 1, 'out_shape': (2, 3, 4)}, {'axis': -3, 'start': 2, 'out_shape': (3, 2, 4)}, {'axis': -1, 'start': 0, 'out_shape': (4, 2, 3)}, {'axis': -2, 'start': -2, 'out_shape': (2, 3, 4)}, {'axis': 0, 'start': 3, 'out_shape': (3, 4, 2)}, {'axis': 2, 'start': -3, 'out_shape': (4, 2, 3)}, {'axis': 0, 'start': 0, 'out_shape': (2, 3, 4)})
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestRollaxis(testing.FunctionTestCase):
    dtype = numpy.float32

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_backward_options = {}
        self.check_double_backward_options.update({'atol': 0.001, 'rtol': 0.01})

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            return 10
        (x,) = inputs
        y = functions.rollaxis(x, self.axis, self.start)
        return (y,)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y_expect = numpy.rollaxis(x, self.axis, self.start)
        return (y_expect,)

@testing.parameterize({'axis': 3, 'start': 0}, {'axis': -4, 'start': 0}, {'axis': 0, 'start': 4}, {'axis': 0, 'start': -4})
class TestRollaxisInvalidType(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(type_check.InvalidType):
            functions.rollaxis(x, self.axis, self.start)

    def test_type_error_cpu(self):
        if False:
            while True:
                i = 10
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_type_error(cuda.to_gpu(self.x))

class TestRollaxisInvalidTypeError(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_axis(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            functions.rollaxis(self.x, 'a', start=0)

    def test_invalid_start(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            functions.rollaxis(self.x, 0, start='a')
testing.run_module(__name__, __file__)