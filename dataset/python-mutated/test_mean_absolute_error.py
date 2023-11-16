import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer import utils
from chainer.testing import attr
from chainer.utils import type_check

@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
@testing.parameterize({'dtype': numpy.float16}, {'dtype': numpy.float32})
class TestMeanAbsoluteError(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.001, 'rtol': 0.001})
            self.check_backward_options.update({'atol': 0.05, 'rtol': 0.05})
            self.check_double_backward_options.update({'atol': 0.05, 'rtol': 0.05})

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        dtype = self.dtype
        x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        diff = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        diff[abs(diff) < 0.01] = 0.5
        x1 = x0 + diff
        return (x0, x1)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x0, x1) = inputs
        loss = functions.mean_absolute_error(x0, x1)
        return (loss,)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x0, x1) = inputs
        loss = 0.0
        for i in numpy.ndindex(x0.shape):
            loss += numpy.abs(x0[i] - x1[i])
        loss /= x0.size
        loss = utils.force_array(loss).astype(x0.dtype)
        return (loss,)

class TestMeanAbsoluteErrorTypeCheck(unittest.TestCase):

    def test_invalid_dtype1(self):
        if False:
            return 10
        x0 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        x1 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_absolute_error(x0, x1)

    def test_invalid_dtype2(self):
        if False:
            while True:
                i = 10
        x0 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32))
        x1 = chainer.Variable(numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float16))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_absolute_error(x0, x1)

class TestMeanAbsoluteErrorFP16Overflow(unittest.TestCase):

    def check_fp16_overflow(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x0 = chainer.Variable(xp.full(shape=(64, 1, 16, 16), fill_value=2, dtype=xp.float16))
        x1 = chainer.Variable(xp.full(shape=(64, 1, 16, 16), fill_value=-2, dtype=xp.float16))
        loss = functions.mean_absolute_error(x0, x1)
        self.assertFalse(xp.isinf(loss.array))

    def test_fp16_overflow_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_fp16_overflow(numpy)

    @attr.gpu
    def test_fp16_overflow_gpu(self):
        if False:
            return 10
        self.check_fp16_overflow(cuda.cupy)
testing.run_module(__name__, __file__)