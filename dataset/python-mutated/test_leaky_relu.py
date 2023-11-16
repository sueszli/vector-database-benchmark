import random
import numpy
from chainer import functions
from chainer import testing

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'slope': ['random', 0.0]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestLeakyReLU(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        if self.slope == 'random':
            self.slope = random.random()
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.0001, 'rtol': 0.001}
            self.check_backward_options = {'atol': 0.0005, 'rtol': 0.005}
            self.check_double_backward_options = {'atol': 0.005, 'rtol': 0.05}

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        return (functions.leaky_relu(x, slope=self.slope),)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        expected = numpy.where(x >= 0, x, x * self.slope)
        return (expected.astype(self.dtype),)
testing.run_module(__name__, __file__)