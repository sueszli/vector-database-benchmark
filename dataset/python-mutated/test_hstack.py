import unittest
import numpy
import six
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check

@testing.parameterize(*testing.product_dict([{'shape': (2, 3, 4), 'y_shape': (2, 6, 4), 'xs_length': 2}, {'shape': (3, 4), 'y_shape': (3, 8), 'xs_length': 2}, {'shape': 3, 'y_shape': (6,), 'xs_length': 2}, {'shape': (), 'y_shape': (2,), 'xs_length': 2}, {'shape': (2, 3, 4), 'y_shape': (2, 3, 4), 'xs_length': 1}, {'shape': (3, 4), 'y_shape': (3, 4), 'xs_length': 1}, {'shape': 3, 'y_shape': (3,), 'xs_length': 1}, {'shape': (), 'y_shape': (1,), 'xs_length': 1}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestHstack(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            return 10
        return tuple([numpy.random.uniform(-1, 1, self.shape).astype(self.dtype) for i in six.moves.range(self.xs_length)])

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        y = functions.hstack(inputs)
        return (y,)

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        y = numpy.hstack(inputs)
        return (y,)

@testing.parameterize({'a_shape': (2, 4, 5), 'b_shape': (3, 4, 5), 'valid': False}, {'a_shape': (3, 4, 6), 'b_shape': (3, 4, 5), 'valid': False}, {'a_shape': (3, 6, 5), 'b_shape': (3, 4, 5), 'valid': True}, {'a_shape': (3, 4), 'b_shape': (4, 4), 'valid': False}, {'a_shape': (3, 4), 'b_shape': (3, 3), 'valid': True}, {'a_shape': (3,), 'b_shape': (4,), 'valid': True}, {'a_shape': 3, 'b_shape': (3, 3), 'valid': False})
class TestHstackTypeCheck(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.xs = [numpy.random.uniform(-1, 1, self.a_shape).astype(numpy.float32), numpy.random.uniform(-1, 1, self.b_shape).astype(numpy.float32)]

    def check_value_check(self):
        if False:
            while True:
                i = 10
        if self.valid:
            functions.hstack(self.xs)
        else:
            with self.assertRaises(type_check.InvalidType):
                functions.hstack(self.xs)

    def test_value_check_cpu(self):
        if False:
            return 10
        self.check_value_check()

    @attr.gpu
    def test_value_check_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_value_check()
testing.run_module(__name__, __file__)