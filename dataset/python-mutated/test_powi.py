import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase

def cu_mat_power(A, power, power_A):
    if False:
        print('Hello World!')
    (y, x) = cuda.grid(2)
    (m, n) = power_A.shape
    if x >= n or y >= m:
        return
    power_A[y, x] = math.pow(A[y, x], int32(power))

def cu_mat_power_binop(A, power, power_A):
    if False:
        return 10
    (y, x) = cuda.grid(2)
    (m, n) = power_A.shape
    if x >= n or y >= m:
        return
    power_A[y, x] = A[y, x] ** power

def vec_pow(r, x, y):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    if i < len(r):
        r[i] = pow(x[i], y[i])

def vec_pow_binop(r, x, y):
    if False:
        return 10
    i = cuda.grid(1)
    if i < len(r):
        r[i] = x[i] ** y[i]

def vec_pow_inplace_binop(r, x):
    if False:
        i = 10
        return i + 15
    i = cuda.grid(1)
    if i < len(r):
        r[i] **= x[i]

def random_complex(N):
    if False:
        while True:
            i = 10
    np.random.seed(123)
    return np.random.random(1) + np.random.random(1) * 1j

class TestCudaPowi(CUDATestCase):

    def test_powi(self):
        if False:
            i = 10
            return i + 15
        dec = cuda.jit(void(float64[:, :], int8, float64[:, :]))
        kernel = dec(cu_mat_power)
        power = 2
        A = np.arange(10, dtype=np.float64).reshape(2, 5)
        Aout = np.empty_like(A)
        kernel[1, A.shape](A, power, Aout)
        self.assertTrue(np.allclose(Aout, A ** power))

    def test_powi_binop(self):
        if False:
            i = 10
            return i + 15
        dec = cuda.jit(void(float64[:, :], int8, float64[:, :]))
        kernel = dec(cu_mat_power_binop)
        power = 2
        A = np.arange(10, dtype=np.float64).reshape(2, 5)
        Aout = np.empty_like(A)
        kernel[1, A.shape](A, power, Aout)
        self.assertTrue(np.allclose(Aout, A ** power))

    def _test_cpow(self, dtype, func, rtol=1e-07):
        if False:
            return 10
        N = 32
        x = random_complex(N).astype(dtype)
        y = random_complex(N).astype(dtype)
        r = np.zeros_like(x)
        cfunc = cuda.jit(func)
        cfunc[1, N](r, x, y)
        np.testing.assert_allclose(r, x ** y, rtol=rtol)
        x = np.asarray([0j, 1j], dtype=dtype)
        y = np.asarray([0j, 1.0], dtype=dtype)
        r = np.zeros_like(x)
        cfunc[1, 2](r, x, y)
        np.testing.assert_allclose(r, x ** y, rtol=rtol)

    def test_cpow_complex64_pow(self):
        if False:
            print('Hello World!')
        self._test_cpow(np.complex64, vec_pow, rtol=3e-07)

    def test_cpow_complex64_binop(self):
        if False:
            print('Hello World!')
        self._test_cpow(np.complex64, vec_pow_binop, rtol=3e-07)

    def test_cpow_complex128_pow(self):
        if False:
            print('Hello World!')
        self._test_cpow(np.complex128, vec_pow)

    def test_cpow_complex128_binop(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_cpow(np.complex128, vec_pow_binop)

    def _test_cpow_inplace_binop(self, dtype, rtol=1e-07):
        if False:
            i = 10
            return i + 15
        N = 32
        x = random_complex(N).astype(dtype)
        y = random_complex(N).astype(dtype)
        r = x ** y
        cfunc = cuda.jit(vec_pow_inplace_binop)
        cfunc[1, N](x, y)
        np.testing.assert_allclose(x, r, rtol=rtol)

    def test_cpow_complex64_inplace_binop(self):
        if False:
            return 10
        self._test_cpow_inplace_binop(np.complex64, rtol=3e-07)

    def test_cpow_complex128_inplace_binop(self):
        if False:
            i = 10
            return i + 15
        self._test_cpow_inplace_binop(np.complex128, rtol=3e-07)
if __name__ == '__main__':
    unittest.main()