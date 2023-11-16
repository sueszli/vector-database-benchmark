import random
import numpy
from chainer import functions
from chainer import testing

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestSELU(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def setUp(self):
        if False:
            print('Hello World!')
        self.alpha = random.random()
        self.scale = random.random()
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.0005, 'rtol': 0.005})
            self.check_backward_options.update({'atol': 0.0005, 'rtol': 0.005})

    def generate_inputs(self):
        if False:
            return 10
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        return (functions.selu(x, alpha=self.alpha, scale=self.scale),)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x,) = inputs
        expected = numpy.where(x >= 0, x, self.alpha * (numpy.exp(x) - 1))
        expected *= self.scale
        return (expected.astype(x.dtype),)
testing.run_module(__name__, __file__)