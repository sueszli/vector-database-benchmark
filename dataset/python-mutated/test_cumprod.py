import unittest
import numpy
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import force_array
from chainer.utils import type_check

@testing.parameterize(*testing.product_dict([{'shape': (1,), 'axis': 0}, {'shape': (2, 3, 4), 'axis': 0}, {'shape': (2, 3, 4), 'axis': 1}, {'shape': (2, 3, 4), 'axis': 2}, {'shape': (2, 3, 4), 'axis': -3}, {'shape': (2, 3, 4), 'axis': -2}, {'shape': (2, 3, 4), 'axis': -1}, {'shape': (2, 3, 4), 'axis': None}], testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contain_zero': [True, False]})) + testing.product({'shape': [(0, 3)], 'axis': [-2, 1, None], 'dtype': [numpy.float64], 'contain_zero': [False]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestCumprod(testing.FunctionTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.01, 'rtol': 0.01})
            self.check_backward_options.update({'atol': 0.01, 'rtol': 0.01})
            self.check_double_backward_options.update({'atol': 0.1, 'rtol': 0.1, 'eps': 0.01})
        elif self.dtype == numpy.float32:
            self.check_double_backward_options.update({'atol': 0.001, 'rtol': 0.001})

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        x = numpy.random.uniform(-2, 2, self.shape).astype(self.dtype)
        if self.contain_zero:
            index = numpy.random.choice(x.size)
            x.ravel()[index] = 0
        return (x,)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        return (functions.cumprod(x, axis=self.axis),)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        expected = numpy.cumprod(x, axis=self.axis)
        expected = force_array(expected)
        return (expected,)

@testing.parameterize({'axis': 3}, {'axis': -4})
class TestCumprodInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(type_check.InvalidType):
            functions.cumprod(x, self.axis)

    def test_type_error_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            print('Hello World!')
        self.check_type_error(cuda.to_gpu(self.x))

class TestCumprodInvalidTypeError(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_type_axis(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, [0])
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, (0,))
testing.run_module(__name__, __file__)