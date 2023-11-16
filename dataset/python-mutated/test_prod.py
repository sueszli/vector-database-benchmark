import unittest
import numpy
from chainer import functions
from chainer import testing

@testing.parameterize(*testing.product({'axis': [None, 0, 1, 2, -1, (0, 1), (1, 0), (0, -1), (-2, 0)], 'keepdims': [True, False], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'contain_zero': [True, False], 'shape': [(3, 2, 4)]}) + testing.product({'axis': [None, 0, 1, 2, (0, 1), (0, -1)], 'keepdims': [True, False], 'dtype': [numpy.float32], 'contain_zero': [False], 'shape': [(3, 1, 0)]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestProd(testing.FunctionTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.001, 'rtol': 0.001})
            self.check_backward_options.update({'atol': 0.001, 'rtol': 0.001})
            self.check_double_backward_options.update({'atol': 0.001, 'rtol': 0.001})

    def generate_inputs(self):
        if False:
            print('Hello World!')
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.contain_zero:
            index = numpy.random.choice(x.size)
            x.ravel()[index] = 0
        return (x,)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x,) = inputs
        y = functions.prod(x, axis=self.axis, keepdims=self.keepdims)
        return (y,)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x,) = inputs
        expected = x.prod(axis=self.axis, keepdims=self.keepdims)
        expected = numpy.asarray(expected)
        return (expected,)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestProdError(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

    def test_invalid_axis_type(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            functions.prod(self.x, axis=[0])

    def test_invalid_axis_type_in_tuple(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            functions.prod(self.x, axis=(1, 'x'))

    def test_duplicate_axis(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            functions.prod(self.x, axis=(0, 0))

    def test_pos_neg_duplicate_axis(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            self.x.prod(axis=(1, -2))
testing.run_module(__name__, __file__)