import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
pi = math.pi

def sinc(x):
    if False:
        for i in range(10):
            print('nop')
    if x == 0.0:
        return 1.0
    else:
        return math.sin(x * pi) / (pi * x)

def scaled_sinc(x, scale):
    if False:
        i = 10
        return i + 15
    if x == 0.0:
        return scale
    else:
        return scale * (math.sin(x * pi) / (pi * x))

def vector_add(a, b):
    if False:
        while True:
            i = 10
    return a + b

class BaseVectorizeDecor(object):
    target = None
    wrapper = None
    funcs = {'func1': sinc, 'func2': scaled_sinc, 'func3': vector_add}

    @classmethod
    def _run_and_compare(cls, func, sig, A, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if cls.wrapper is not None:
            func = cls.wrapper(func)
        numba_func = vectorize(sig, target=cls.target)(func)
        numpy_func = np.vectorize(func)
        result = numba_func(A, *args)
        gold = numpy_func(A, *args)
        np.testing.assert_allclose(result, gold, **kwargs)

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        sig = ['float64(float64)', 'float32(float32)']
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_2(self):
        if False:
            print('Hello World!')
        sig = [float64(float64), float32(float32)]
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        sig = ['float64(float64, uint32)']
        func = self.funcs['func2']
        A = np.arange(100, dtype=np.float64)
        scale = np.uint32(3)
        self._run_and_compare(func, sig, A, scale, atol=1e-08)

    def test_4(self):
        if False:
            for i in range(10):
                print('nop')
        sig = [int32(int32, int32), uint32(uint32, uint32), float32(float32, float32), float64(float64, float64)]
        func = self.funcs['func3']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.float32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.int32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.uint32)
        self._run_and_compare(func, sig, A, A)

class TestCPUVectorizeDecor(unittest.TestCase, BaseVectorizeDecor):
    target = 'cpu'

class TestParallelVectorizeDecor(unittest.TestCase, BaseVectorizeDecor):
    _numba_parallel_test_ = False
    target = 'parallel'

class TestCPUVectorizeJitted(unittest.TestCase, BaseVectorizeDecor):
    target = 'cpu'
    wrapper = jit(nopython=True)

class BaseVectorizeNopythonArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing the nopython argument to the vectorize decorator.
    """

    def _test_target_nopython(self, target, warnings, with_sig=True):
        if False:
            i = 10
            return i + 15
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.check_warnings(warnings):
            f = vectorize(*args, target=target, nopython=True)(vector_add)
            f(a, b)

class TestVectorizeNopythonArg(BaseVectorizeNopythonArg):

    def test_target_cpu_nopython(self):
        if False:
            print('Hello World!')
        self._test_target_nopython('cpu', [])

    def test_target_cpu_nopython_no_sig(self):
        if False:
            i = 10
            return i + 15
        self._test_target_nopython('cpu', [], False)

    def test_target_parallel_nopython(self):
        if False:
            return 10
        self._test_target_nopython('parallel', [])

class BaseVectorizeUnrecognizedArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing an unrecognized argument to the vectorize decorator.
    """

    def _test_target_unrecognized_arg(self, target, with_sig=True):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.assertRaises(KeyError) as raises:
            f = vectorize(*args, target=target, nonexistent=2)(vector_add)
            f(a, b)
        self.assertIn('Unrecognized options', str(raises.exception))

class TestVectorizeUnrecognizedArg(BaseVectorizeUnrecognizedArg):

    def test_target_cpu_unrecognized_arg(self):
        if False:
            i = 10
            return i + 15
        self._test_target_unrecognized_arg('cpu')

    def test_target_cpu_unrecognized_arg_no_sig(self):
        if False:
            print('Hello World!')
        self._test_target_unrecognized_arg('cpu', False)

    def test_target_parallel_unrecognized_arg(self):
        if False:
            while True:
                i = 10
        self._test_target_unrecognized_arg('parallel')
if __name__ == '__main__':
    unittest.main()