import math
import unittest
import numpy
from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainer import utils

def _ndtr_cpu(x, dtype):
    if False:
        while True:
            i = 10
    erfc = numpy.vectorize(lambda x: 0.5 * math.erfc(-x / 2 ** 0.5))
    return utils.force_array(erfc(x), dtype=dtype)

def _ndtr_gpu(x, dtype):
    if False:
        return 10
    return cuda.to_gpu(_ndtr_cpu(cuda.to_cpu(x), dtype))

def _ndtr_expected(x, dtype):
    if False:
        print('Hello World!')
    if backend.get_array_module(x) is numpy:
        return _ndtr_cpu(x, dtype)
    else:
        return _ndtr_gpu(x, dtype)

@testing.unary_math_function_unittest(F.ndtr, func_expected=_ndtr_expected)
class TestNdtr(unittest.TestCase):
    pass
testing.run_module(__name__, __file__)