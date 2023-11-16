import numpy
import six
from chainer import functions
from chainer import testing
from chainer.testing import backend

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@backend.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestLocalResponseNormalization(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        self.skip_double_backward_test = True
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.0001, 'rtol': 0.001}
            self.check_backward_options = {'atol': 0.005, 'rtol': 0.005}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 0.0003, 'rtol': 0.003}

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        x = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(self.dtype)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        y_expect = numpy.zeros_like(x)
        for (n, c, h, w) in numpy.ndindex(x.shape):
            s = 0
            for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
                s += x[n, i, h, w] ** 2
            denom = (2 + 0.0001 * s) ** 0.75
            y_expect[n, c, h, w] = x[n, c, h, w] / denom
        return (y_expect,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y = functions.local_response_normalization(x)
        return (y,)
testing.run_module(__name__, __file__)