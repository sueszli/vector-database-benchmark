from numba import vectorize, jit, bool_, double, int_, float_, typeof, int8
import unittest
import numpy as np

def add(a, b):
    if False:
        print('Hello World!')
    return a + b

def func(dtypeA, dtypeB):
    if False:
        while True:
            i = 10
    A = np.arange(10, dtype=dtypeA)
    B = np.arange(10, dtype=dtypeB)
    return typeof(vector_add(A, B))

class TestVectTypeInfer(unittest.TestCase):

    def test_type_inference(self):
        if False:
            return 10
        'This is testing numpy ufunc dispatch machinery\n        '
        global vector_add
        vector_add = vectorize([bool_(double, int_), double(double, double), float_(double, float_)])(add)
        cfunc = jit(func)

        def numba_type_equal(a, b):
            if False:
                print('Hello World!')
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.ndim, b.ndim)
        numba_type_equal(cfunc(np.dtype(np.float64), np.dtype('i')), bool_[:])
        numba_type_equal(cfunc(np.dtype(np.float64), np.dtype(np.float64)), double[:])
        numba_type_equal(cfunc(np.dtype(np.float64), np.dtype(np.float32)), double[:])
if __name__ == '__main__':
    unittest.main()