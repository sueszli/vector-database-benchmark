import math
import numpy
from chainer import functions
from chainer import testing

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestFmod(testing.FunctionTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.check_forward_options.update({'atol': 1e-07, 'rtol': 1e-07})
        self.check_backward_options.update({'atol': 0.0005, 'rtol': 0.005})
        self.check_double_backward_options.update({'atol': 0.001, 'rtol': 0.01})

    def generate_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = numpy.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        divisor = numpy.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if math.fabs(divisor[i]) < 0.1:
                divisor[i] += 1.0
        for i in numpy.ndindex(self.shape):
            m = math.fabs(x[i] % divisor[i])
            if m < 0.01 or m > divisor[i] - 0.01:
                x[i] = 0.5
                divisor[i] = 0.3
        return (x, divisor)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (x, divisor) = inputs
        y = functions.fmod(x, divisor)
        return (y,)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x, divisor) = inputs
        expected = numpy.fmod(x, divisor)
        expected = numpy.asarray(expected)
        return (expected,)
testing.run_module(__name__, __file__)