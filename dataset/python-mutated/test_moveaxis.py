import unittest
import numpy
import six
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check

def _normalize_axis_tuple(axis, ndim):
    if False:
        for i in range(10):
            print('nop')
    if numpy.isscalar(axis):
        axis = (axis,)
    ret = []
    for ax in axis:
        ret.append(ax % ndim)
    return ret

def _moveaxis(a, source, destination):
    if False:
        i = 10
        return i + 15
    if hasattr(numpy, 'moveaxis'):
        return numpy.moveaxis(a, source, destination)
    source = _normalize_axis_tuple(source, a.ndim)
    destination = _normalize_axis_tuple(destination, a.ndim)
    order = [n for n in six.moves.range(a.ndim) if n not in source]
    for (dest, src) in sorted(six.moves.zip(destination, source)):
        order.insert(dest, src)
    result = a.transpose(order)
    return result

@testing.parameterize({'source': 0, 'destination': -1, 'out_shape': (3, 4, 2)}, {'source': -1, 'destination': 1, 'out_shape': (2, 4, 3)}, {'source': (0, 2), 'destination': (1, 0), 'out_shape': (4, 2, 3)}, {'source': (0, -1), 'destination': (-1, 1), 'out_shape': (3, 4, 2)})
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestMoveaxis(testing.FunctionTestCase):
    dtype = numpy.float32

    def setUp(self):
        if False:
            while True:
                i = 10
        self.check_backward_options = {}
        self.check_double_backward_options = {'atol': 0.001, 'rtol': 0.01}

    def generate_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(self.dtype)
        return (x,)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x,) = inputs
        y = functions.moveaxis(x, self.source, self.destination)
        return (y,)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        y_expect = _moveaxis(x, self.source, self.destination)
        return (y_expect,)

@testing.parameterize({'source': 4, 'destination': 0}, {'source': 0, 'destination': 4}, {'source': 0, 'destination': -4}, {'source': -4, 'destination': 0})
class TestMoveaxisInvalidType(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            while True:
                i = 10
        with self.assertRaises(type_check.InvalidType):
            functions.moveaxis(x, self.source, self.destination)

    def test_type_error_cpu(self):
        if False:
            return 10
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            while True:
                i = 10
        self.check_type_error(cuda.to_gpu(self.x))

@testing.parameterize({'source': (1, 2), 'destination': (1, 2, 0)}, {'source': (0, 0), 'destination': (1, 2)}, {'source': (0, 1), 'destination': (2, 2)})
class TestMoveaxisValueError(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            functions.moveaxis(x, self.source, self.destination)

    def test_type_error_cpu(self):
        if False:
            return 10
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_type_error(cuda.to_gpu(self.x))

@testing.parameterize({'source': (1, 2), 'destination': (1, 2.0)}, {'source': (1, 2.0), 'destination': (1, 2)})
class TestMoveaxisTypeError(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            functions.moveaxis(x, self.source, self.destination)

    @testing.with_requires('numpy!=1.11.*')
    def test_type_error_cpu(self):
        if False:
            return 10
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        if False:
            return 10
        self.check_type_error(cuda.to_gpu(self.x))
testing.run_module(__name__, __file__)