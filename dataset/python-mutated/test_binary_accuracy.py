import unittest
import numpy
import six
import chainer
from chainer import functions
from chainer import testing
from chainer.utils import force_array
from chainer.utils import type_check

@testing.parameterize(*testing.product({'shape': [(9, 11), (99,)], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'label_dtype': [numpy.int8, numpy.int16, numpy.int32, numpy.int64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestBinaryAccuracy(testing.FunctionTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.skip_backward_test = True
        self.skip_double_backward_test = True
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.0001, 'rtol': 0.001})

    def generate_inputs(self):
        if False:
            return 10
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        t = numpy.random.randint(-1, 2, self.shape).astype(self.label_dtype)
        return (x, t)

    def forward(self, inputs, device):
        if False:
            while True:
                i = 10
        (x, t) = inputs
        return (functions.binary_accuracy(x, t),)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x, t) = inputs
        count = 0
        correct = 0
        x_flatten = x.ravel()
        t_flatten = t.ravel()
        for i in six.moves.range(t_flatten.size):
            if t_flatten[i] == -1:
                continue
            pred = int(x_flatten[i] >= 0)
            if pred == t_flatten[i]:
                correct += 1
            count += 1
        expected = float(correct) / count
        expected = force_array(expected, self.dtype)
        return (expected,)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.fix_random()
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestBinaryAccuracyIgnoreAll(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        self.skip_backward_test = True
        self.skip_double_backward_test = True

    def generate_inputs(self):
        if False:
            return 10
        shape = (5, 4)
        x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        t = -numpy.ones(shape).astype(numpy.int32)
        return (x, t)

    def forward(self, inputs, device):
        if False:
            return 10
        (x, t) = inputs
        return (functions.binary_accuracy(x, t),)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        return (force_array(0.0, self.dtype),)

class TestBinaryAccuracyTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        if False:
            print('Hello World!')
        x = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 3, 5), dtype=numpy.int32))
        with self.assertRaises(type_check.InvalidType):
            chainer.functions.binary_accuracy(x, t)

    def test_invalid_type(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))
        t = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))
        with self.assertRaises(type_check.InvalidType):
            chainer.functions.binary_accuracy(x, t)
testing.run_module(__name__, __file__)