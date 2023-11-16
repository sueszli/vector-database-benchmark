import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout

@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestVecAdd(CUDATestCase):
    """
    Test simple vector addition
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_vecadd(self):
        if False:
            print('Hello World!')
        import numpy as np
        from numba import cuda

        @cuda.jit
        def f(a, b, c):
            if False:
                return 10
            tid = cuda.grid(1)
            size = len(c)
            if tid < size:
                c[tid] = a[tid] + b[tid]
        np.random.seed(1)
        N = 100000
        a = cuda.to_device(np.random.random(N))
        b = cuda.to_device(np.random.random(N))
        c = cuda.device_array_like(a)
        f.forall(len(a))(a, b, c)
        print(c.copy_to_host())
        nthreads = 256
        nblocks = len(a) // nthreads + 1
        f[nblocks, nthreads](a, b, c)
        print(c.copy_to_host())
        np.testing.assert_equal(c.copy_to_host(), a.copy_to_host() + b.copy_to_host())
if __name__ == '__main__':
    unittest.main()