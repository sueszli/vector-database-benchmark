"""
There was a deadlock problem when work count is smaller than number of threads.
"""
import numpy as np
from numba import float32, float64, int32, uint32
from numba.np.ufunc import Vectorize
import unittest

def vector_add(a, b):
    if False:
        return 10
    return a + b

class TestParallelLowWorkCount(unittest.TestCase):
    _numba_parallel_test_ = False

    def test_low_workcount(self):
        if False:
            for i in range(10):
                print('nop')
        pv = Vectorize(vector_add, target='parallel')
        for ty in (int32, uint32, float32, float64):
            pv.add(ty(ty, ty))
        para_ufunc = pv.build_ufunc()
        np_ufunc = np.vectorize(vector_add)

        def test(ty):
            if False:
                for i in range(10):
                    print('nop')
            data = np.arange(1).astype(ty)
            result = para_ufunc(data, data)
            gold = np_ufunc(data, data)
            np.testing.assert_allclose(gold, result)
        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)
if __name__ == '__main__':
    unittest.main()