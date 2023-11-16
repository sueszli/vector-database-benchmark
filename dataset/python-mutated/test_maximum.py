import unittest
import numpy
import chainer
from chainer import functions
from chainer import testing
from chainer.utils import type_check

@testing.parameterize(*testing.product({'shape': [((3, 2), (3, 2), (3, 2)), ((), (), ()), ((3, 2), (3, 1), (3, 2)), ((2,), (3, 2), (3, 2))], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestMaximum(testing.FunctionTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.dtype == numpy.float16:
            eps = 0.01
            self.check_forward_options.update({'atol': 0.0001, 'rtol': 0.001})
            self.check_backward_options.update({'atol': 0.01, 'rtol': 0.01})
            self.check_double_backward_options.update({'atol': 0.01, 'rtol': 0.01})
        else:
            eps = 0.001
        self.check_backward_options['eps'] = eps
        self.check_double_backward_options['eps'] = eps

    def generate_inputs(self):
        if False:
            print('Hello World!')
        (x1_shape, x2_shape, y_shape) = self.shape
        x1 = numpy.random.uniform(-1, 1, x1_shape).astype(self.dtype)
        x2 = numpy.random.uniform(-1, 1, x2_shape).astype(self.dtype)
        return (x1, x2)

    def forward(self, inputs, devices):
        if False:
            print('Hello World!')
        (x1, x2) = inputs
        return (functions.maximum(x1, x2),)

    def forward_expected(self, inputs):
        if False:
            while True:
                i = 10
        (x1, x2) = inputs
        expected = numpy.maximum(x1, x2)
        expected = numpy.asarray(expected)
        return (expected,)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestMaximumInconsistentShapes(unittest.TestCase):

    def test_maximum_inconsistent_shapes(self):
        if False:
            print('Hello World!')
        x1_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)
        x2_data = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with self.assertRaises(type_check.InvalidType):
            functions.maximum(x1, x2)
testing.run_module(__name__, __file__)