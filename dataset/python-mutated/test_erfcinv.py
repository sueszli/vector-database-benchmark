import unittest
import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing

def _erfcinv_cpu(x, dtype):
    if False:
        while True:
            i = 10
    from scipy import special
    return numpy.vectorize(special.erfcinv, otypes=[dtype])(x)

def _erfcinv_gpu(x, dtype):
    if False:
        return 10
    return cuda.to_gpu(_erfcinv_cpu(cuda.to_cpu(x), dtype))

def _erfcinv_expected(x, dtype):
    if False:
        i = 10
        return i + 15
    if backend.get_array_module(x) is numpy:
        return _erfcinv_cpu(x, dtype)
    else:
        return _erfcinv_gpu(x, dtype)

def make_data(shape, dtype):
    if False:
        i = 10
        return i + 15
    x = numpy.random.uniform(0.1, 1.9, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return (x, gy, ggx)

@testing.unary_math_function_unittest(F.erfcinv, func_expected=_erfcinv_expected, make_data=make_data, forward_options={'atol': 0.001, 'rtol': 0.001}, backward_options={'eps': 1e-06}, double_backward_options={'eps': 1e-06})
@testing.with_requires('scipy')
class TestErfcinv(unittest.TestCase):
    pass

@testing.parameterize(*testing.product({'shape': [(3, 2), ()], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.without_requires('scipy')
class TestErfcinvExceptions(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        (self.x, self.gy, self.ggx) = make_data(self.shape, self.dtype)
        self.func = F.erfcinv

    def check_forward(self, x_data):
        if False:
            return 10
        x = chainer.Variable(x_data)
        with self.assertRaises(ImportError):
            self.func(x)

    def test_forward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_forward(self.x)
testing.run_module(__name__, __file__)