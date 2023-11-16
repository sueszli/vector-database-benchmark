import unittest
from numba.tests.support import captured_stdout, skip_parfors_unsupported
from numba import set_parallel_chunksize
from numba.tests.support import TestCase

@skip_parfors_unsupported
class ChunksizeExamplesTest(TestCase):
    _numba_parallel_test_ = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        set_parallel_chunksize(0)

    def tearDown(self):
        if False:
            while True:
                i = 10
        set_parallel_chunksize(0)

    def test_unbalanced_example(self):
        if False:
            while True:
                i = 10
        with captured_stdout():
            from numba import njit, prange
            import numpy as np

            @njit(parallel=True)
            def func1():
                if False:
                    print('Hello World!')
                n = 100
                vals = np.empty(n)
                for i in prange(n):
                    cur = i + 1
                    for j in range(i):
                        if cur % 2 == 0:
                            cur //= 2
                        else:
                            cur = cur * 3 + 1
                    vals[i] = cur
                return vals
            result = func1()
            self.assertPreciseEqual(result, func1.py_func())

    def test_chunksize_manual(self):
        if False:
            while True:
                i = 10
        with captured_stdout():
            from numba import njit, prange, set_parallel_chunksize, get_parallel_chunksize

            @njit(parallel=True)
            def func1(n):
                if False:
                    i = 10
                    return i + 15
                acc = 0
                print(get_parallel_chunksize())
                for i in prange(n):
                    print(get_parallel_chunksize())
                    acc += i
                print(get_parallel_chunksize())
                return acc

            @njit(parallel=True)
            def func2(n):
                if False:
                    for i in range(10):
                        print('nop')
                acc = 0
                old_chunksize = get_parallel_chunksize()
                set_parallel_chunksize(8)
                for i in prange(n):
                    acc += i
                set_parallel_chunksize(old_chunksize)
                return acc
            old_chunksize = set_parallel_chunksize(4)
            result1 = func1(12)
            result2 = func2(12)
            result3 = func1(12)
            set_parallel_chunksize(old_chunksize)
            self.assertPreciseEqual(result1, func1.py_func(12))
            self.assertPreciseEqual(result2, func2.py_func(12))
            self.assertPreciseEqual(result3, func1.py_func(12))

    def test_chunksize_with(self):
        if False:
            return 10
        with captured_stdout():
            from numba import njit, prange, parallel_chunksize

            @njit(parallel=True)
            def func1(n):
                if False:
                    i = 10
                    return i + 15
                acc = 0
                for i in prange(n):
                    acc += i
                return acc

            @njit(parallel=True)
            def func2(n):
                if False:
                    i = 10
                    return i + 15
                acc = 0
                with parallel_chunksize(8):
                    for i in prange(n):
                        acc += i
                return acc
            with parallel_chunksize(4):
                result1 = func1(12)
                result2 = func2(12)
                result3 = func1(12)
            self.assertPreciseEqual(result1, func1.py_func(12))
            self.assertPreciseEqual(result2, func2.py_func(12))
            self.assertPreciseEqual(result3, func1.py_func(12))
if __name__ == '__main__':
    unittest.main()