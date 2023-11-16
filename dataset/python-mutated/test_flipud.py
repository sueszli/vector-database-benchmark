import numpy
from chainer import functions
from chainer import testing

@testing.parameterize(*testing.product({'shape': [(3,), (3, 4)], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestFlipUD(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            return 10
        x = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x,) = inputs
        y = numpy.flipud(x)
        return (y,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        return (functions.flipud(x),)
testing.run_module(__name__, __file__)