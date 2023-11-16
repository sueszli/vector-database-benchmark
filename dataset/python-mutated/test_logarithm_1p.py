import numpy
from chainer import functions
from chainer import testing
from chainer.utils import force_array

@testing.parameterize(*testing.product({'shape': [(), (3, 2)], 'dtype': [numpy.float32]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class Log1pFunctionTest(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = numpy.random.uniform(0.5, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            while True:
                i = 10
        (x,) = inputs
        return (functions.log1p(x),)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        expected = numpy.log1p(x)
        expected = force_array(expected)
        return (expected,)
testing.run_module(__name__, __file__)