import unittest
import numpy
import chainer
from chainer import functions
from chainer import testing
from chainer import utils
from chainer.utils import type_check

@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
@testing.parameterize({'dtype': numpy.float16}, {'dtype': numpy.float32})
class TestMeanSquaredError(testing.FunctionTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.001, 'rtol': 0.001})
            self.check_backward_options.update({'atol': 0.05, 'rtol': 0.05})
            self.check_double_backward_options.update({'atol': 0.05, 'rtol': 0.05})

    def generate_inputs(self):
        if False:
            print('Hello World!')
        dtype = self.dtype
        x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        x1 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        return (x0, x1)

    def forward(self, inputs, device):
        if False:
            return 10
        (x0, x1) = inputs
        loss = functions.mean_squared_error(x0, x1)
        return (loss,)

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x0, x1) = inputs
        loss = 0.0
        for i in numpy.ndindex(x0.shape):
            loss += (x0[i] - x1[i]) ** 2
        loss /= x0.size
        loss = utils.force_array(loss).astype(x0.dtype)
        return (loss,)

class TestMeanSquaredErrorTypeCheck(unittest.TestCase):

    def test_invalid_dtype1(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        x1 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_squared_error(x0, x1)

    def test_invalid_dtype2(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32))
        x1 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float16))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_squared_error(x0, x1)
testing.run_module(__name__, __file__)