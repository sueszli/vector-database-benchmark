import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase

class TestCudaAutoContext(CUDATestCase):

    def test_auto_context(self):
        if False:
            for i in range(10):
                print('nop')
        'A problem was revealed by a customer that the use cuda.to_device\n        does not create a CUDA context.\n        This tests the problem\n        '
        A = np.arange(10, dtype=np.float32)
        newA = np.empty_like(A)
        dA = cuda.to_device(A)
        dA.copy_to_host(newA)
        self.assertTrue(np.allclose(A, newA))
if __name__ == '__main__':
    unittest.main()