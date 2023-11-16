import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer import utils

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'shape': [(), (1,), (1, 1), (4,), (4, 3), (4, 3, 2)]}))
@testing.inject_backend_tests(['test_forward', 'test_backward', 'test_double_backward'], [{}] + testing.product({'use_cuda': [True]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0']}))
class TestAbsoluteError(testing.FunctionTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.001, 'rtol': 0.001})
            self.check_backward_options.update({'atol': 0.05, 'rtol': 0.05})
            self.check_double_backward_options.update({'atol': 0.3, 'rtol': 0.3})

    def generate_inputs(self):
        if False:
            return 10
        x0 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff[abs(diff) < 0.02] = 0.5
        x1 = numpy.asarray(x0 + diff)
        return (x0, x1)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x0, x1) = inputs
        return (utils.force_array(numpy.abs(x0 - x1), self.dtype),)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x0, x1) = inputs
        return (functions.absolute_error(x0, x1),)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'shape': [(), (1,), (1, 1), (4,), (4, 3), (4, 3, 2)]}))
class TestNonDefaultGPU(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x0 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff[abs(diff) < 0.02] = 0.5
        self.x1 = numpy.asarray(self.x0 + diff)
        self.gy = numpy.random.random(self.shape).astype(self.dtype)

    @attr.multi_gpu(2)
    def test_backward_non_default_gpu(self):
        if False:
            print('Hello World!')
        x0 = chainer.Variable(cuda.to_gpu(self.x0, 1))
        x1 = chainer.Variable(cuda.to_gpu(self.x1, 1))
        gy = cuda.to_gpu(self.gy, 1)
        with cuda.get_device_from_id(0):
            y = functions.absolute_error(x0, x1)
            y.grad = gy
            y.backward()
testing.run_module(__name__, __file__)