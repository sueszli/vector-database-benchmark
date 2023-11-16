import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase

class TestFreeVar(CUDATestCase):

    def test_freevar(self):
        if False:
            while True:
                i = 10
        'Make sure we can compile the following kernel with freevar reference\n        in arguments to shared.array\n        '
        from numba import float32
        size = 1024
        nbtype = float32

        @cuda.jit('(float32[::1], intp)')
        def foo(A, i):
            if False:
                i = 10
                return i + 15
            'Dummy function'
            sdata = cuda.shared.array(size, dtype=nbtype)
            A[i] = sdata[i]
        A = np.arange(2, dtype='float32')
        foo[1, 1](A, 0)
if __name__ == '__main__':
    unittest.main()