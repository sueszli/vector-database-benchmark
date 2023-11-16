import unittest
import numpy
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import force_array
from chainer.utils import type_check

@testing.parameterize(*testing.product_dict([{'shape': (1,), 'axis': 0}, {'shape': (2, 3, 4), 'axis': 0}, {'shape': (2, 3, 4), 'axis': 1}, {'shape': (2, 3, 4), 'axis': 2}, {'shape': (2, 3, 4), 'axis': -3}, {'shape': (2, 3, 4), 'axis': -2}, {'shape': (2, 3, 4), 'axis': -1}, {'shape': (2, 3, 4), 'axis': None}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestCumsum(testing.FunctionTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.01})
            self.check_backward_options.update({'atol': 0.01})
            self.check_double_backward_options.update({'atol': 0.1, 'eps': 0.01})
        elif self.dtype == numpy.float32:
            self.check_double_backward_options.update({'atol': 0.001})

    def generate_inputs(self):
        if False:
            i = 10
            return i + 15
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        return (functions.cumsum(x, axis=self.axis),)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        expected = numpy.cumsum(x, axis=self.axis)
        expected = force_array(expected)
        return (expected,)

@testing.parameterize({'axis': 3}, {'axis': -4})
class TestCumsumInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(type_check.InvalidType):
            functions.cumsum(x, self.axis)

    def test_type_error_cpu(self):
        if False:
            print('Hello World!')
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_type_error(cuda.to_gpu(self.x))

class TestCumsumInvalidTypeError(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_type_axis(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            functions.cumsum(self.x, [0])
        with self.assertRaises(TypeError):
            functions.cumsum(self.x, (0,))
testing.run_module(__name__, __file__)