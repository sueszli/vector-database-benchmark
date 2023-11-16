import unittest
import numpy
from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing

def _log_ndtr_cpu(x, dtype):
    if False:
        return 10
    from scipy import special
    return special.log_ndtr(x).astype(dtype)

def _log_ndtr_gpu(x, dtype):
    if False:
        while True:
            i = 10
    return cuda.to_gpu(_log_ndtr_cpu(cuda.to_cpu(x), dtype))

def _log_ndtr_expected(x, dtype):
    if False:
        print('Hello World!')
    if backend.get_array_module(x) is numpy:
        return _log_ndtr_cpu(x, dtype)
    else:
        return _log_ndtr_gpu(x, dtype)

@testing.unary_math_function_unittest(F.log_ndtr, func_expected=_log_ndtr_expected)
@testing.with_requires('scipy')
class TestLogNdtr(unittest.TestCase):
    pass
testing.run_module(__name__, __file__)