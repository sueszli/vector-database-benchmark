import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend

def _decov(h):
    if False:
        i = 10
        return i + 15
    h_mean = h.mean(axis=0)
    (N, M) = h.shape
    loss_expect = numpy.zeros((M, M), dtype=h.dtype)
    for i in six.moves.range(M):
        for j in six.moves.range(M):
            if i != j:
                for n in six.moves.range(N):
                    loss_expect[i, j] += (h[n, i] - h_mean[i]) * (h[n, j] - h_mean[j])
    return loss_expect / N

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'reduce': ['half_squared_sum', 'no']}))
@backend.inject_backend_tests(None, [{}] + [{'use_cuda': True}])
class TestDeCov(testing.FunctionTestCase):
    skip_double_backward_test = True

    def setUp(self):
        if False:
            return 10
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'rtol': 0.01, 'atol': 0.01})
            self.check_backward_options.update({'atol': 0.03, 'eps': 0.02})
        else:
            self.check_forward_options.update({'rtol': 0.0001, 'atol': 0.0001})
            self.check_backward_options.update({'atol': 0.001, 'eps': 0.02})

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        h = numpy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        return (h,)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (h,) = inputs
        loss_expect = _decov(h)
        if self.reduce == 'half_squared_sum':
            loss_expect = (loss_expect ** 2).sum() * 0.5
        return (chainer.utils.force_array(loss_expect, self.dtype),)

    def forward(self, inputs, device):
        if False:
            return 10
        (h,) = inputs
        loss = functions.decov(h, self.reduce)
        return (loss,)

class TestDeconvInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.h = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        if False:
            return 10
        h = xp.asarray(self.h)
        with self.assertRaises(ValueError):
            functions.decov(h, 'invalid_option')

    def test_invalid_option_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        if False:
            while True:
                i = 10
        self.check_invalid_option(cuda.cupy)
testing.run_module(__name__, __file__)