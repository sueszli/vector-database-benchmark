import numpy
from chainer import functions
from chainer import testing
from chainer import utils

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestSoftplus(testing.FunctionTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.beta = numpy.random.uniform(1, 2, ())
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.0005, 'rtol': 0.005}
            self.check_backward_options = {'atol': 0.005, 'rtol': 0.05}
            self.check_double_backward_options = {'atol': 0.05, 'rtol': 0.5}

    def generate_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = numpy.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y = numpy.log(1 + numpy.exp(self.beta * x)) / self.beta
        return (utils.force_array(y).astype(self.dtype),)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x,) = inputs
        return (functions.softplus(x, beta=self.beta),)
testing.run_module(__name__, __file__)