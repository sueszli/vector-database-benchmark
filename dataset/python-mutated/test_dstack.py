import unittest
import numpy
import six
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check

@testing.parameterize(*testing.product_dict([{'shape': (2, 3, 4), 'y_shape': (2, 3, 8), 'xs_length': 2}, {'shape': (3, 4), 'y_shape': (3, 4, 2), 'xs_length': 2}, {'shape': 3, 'y_shape': (1, 3, 2), 'xs_length': 2}, {'shape': (), 'y_shape': (1, 1, 2), 'xs_length': 2}, {'shape': (2, 3, 4), 'y_shape': (2, 3, 4), 'xs_length': 1}, {'shape': (3, 4), 'y_shape': (3, 4, 1), 'xs_length': 1}, {'shape': 3, 'y_shape': (1, 3, 1), 'xs_length': 1}, {'shape': (), 'y_shape': (1, 1, 1), 'xs_length': 1}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestDstack(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 0.0005, 'rtol': 0.005}

    def generate_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        xs = tuple([numpy.random.uniform(-1, 1, self.shape).astype(self.dtype) for i in six.moves.range(self.xs_length)])
        return xs

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        xs = inputs
        y_expected = numpy.dstack(xs)
        return (y_expected,)

    def forward(self, inputs, device):
        if False:
            return 10
        xs = inputs
        y = functions.dstack(xs)
        return (y,)

@testing.parameterize({'a_shape': (2, 3, 4, 6), 'b_shape': (2, 3, 4, 5), 'valid': False}, {'a_shape': (2, 3, 5, 6), 'b_shape': (2, 3, 4, 6), 'valid': True}, {'a_shape': (2, 4, 5), 'b_shape': (3, 4, 5), 'valid': False}, {'a_shape': (3, 4, 6), 'b_shape': (3, 4, 5), 'valid': True}, {'a_shape': (3, 6, 5), 'b_shape': (3, 4, 5), 'valid': False}, {'a_shape': (3, 4), 'b_shape': (4, 4), 'valid': False}, {'a_shape': (3, 4), 'b_shape': (3, 3), 'valid': False}, {'a_shape': (3,), 'b_shape': (4,), 'valid': False}, {'a_shape': 3, 'b_shape': (3, 3), 'valid': False}, {'a_shape': (), 'b_shape': 1, 'valid': False})
class TestDstackTypeCheck(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.xs = [numpy.random.uniform(-1, 1, self.a_shape).astype(numpy.float32), numpy.random.uniform(-1, 1, self.b_shape).astype(numpy.float32)]

    def check_value_check(self):
        if False:
            return 10
        if self.valid:
            functions.dstack(self.xs)
        else:
            with self.assertRaises(type_check.InvalidType):
                functions.dstack(self.xs)

    def test_value_check_cpu(self):
        if False:
            print('Hello World!')
        self.check_value_check()

    @attr.gpu
    def test_value_check_gpu(self):
        if False:
            print('Hello World!')
        self.check_value_check()
testing.run_module(__name__, __file__)