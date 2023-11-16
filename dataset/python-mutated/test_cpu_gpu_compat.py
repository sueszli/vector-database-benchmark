import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
import numpy as np

@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestCpuGpuCompat(CUDATestCase):
    """
    Test compatibility of CPU and GPU functions
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
            return 10
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_cpu_gpu_compat(self):
        if False:
            return 10
        from math import pi
        import numba
        from numba import cuda
        X = cuda.to_device([1, 10, 234])
        Y = cuda.to_device([2, 2, 4014])
        Z = cuda.to_device([3, 14, 2211])
        results = cuda.to_device([0.0, 0.0, 0.0])

        @numba.jit
        def business_logic(x, y, z):
            if False:
                i = 10
                return i + 15
            return 4 * z * (2 * x - 4 * y / 2 * pi)
        print(business_logic(1, 2, 3))

        @cuda.jit
        def f(res, xarr, yarr, zarr):
            if False:
                return 10
            tid = cuda.grid(1)
            if tid < len(xarr):
                res[tid] = business_logic(xarr[tid], yarr[tid], zarr[tid])
        f.forall(len(X))(results, X, Y, Z)
        print(results)
        expect = [business_logic(x, y, z) for (x, y, z) in zip(X, Y, Z)]
        np.testing.assert_equal(expect, results.copy_to_host())
if __name__ == '__main__':
    unittest.main()