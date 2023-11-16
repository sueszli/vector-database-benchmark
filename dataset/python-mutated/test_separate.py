import numpy
from chainer import functions
from chainer import testing
from chainer.utils import force_array

@testing.parameterize(*testing.product_dict([{'shape': (2, 3, 4), 'axis': 0}, {'shape': (2, 3, 4), 'axis': 1}, {'shape': (2, 3, 4), 'axis': 2}, {'shape': (2, 3, 4), 'axis': -1}, {'shape': (2, 3, 4), 'axis': -3}, {'shape': (2,), 'axis': 0}, {'shape': (2,), 'axis': -1}], [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': numpy.float64}]))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestSeparate(testing.FunctionTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.skip_double_backward_test = True

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        return functions.separate(x, self.axis)

    def forward_expected(self, inputs):
        if False:
            return 10
        (x,) = inputs
        return tuple((force_array(x.take(i, axis=self.axis)) for i in range(self.shape[self.axis])))
testing.run_module(__name__, __file__)