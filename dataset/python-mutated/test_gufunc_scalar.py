"""Example: sum each row using guvectorize

See Numpy documentation for detail about gufunc:
    http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
"""
import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest

@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestGUFuncScalar(CUDATestCase):

    def test_gufunc_scalar_output(self):
        if False:
            for i in range(10):
                print('nop')

        @guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='cuda')
        def sum_row(inp, out):
            if False:
                while True:
                    i = 10
            tmp = 0.0
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[0] = tmp
        inp = np.arange(300, dtype=np.int32).reshape(100, 3)
        out1 = np.empty(100, dtype=inp.dtype)
        out2 = np.empty(100, dtype=inp.dtype)
        dev_inp = cuda.to_device(inp)
        dev_out1 = cuda.to_device(out1, copy=False)
        sum_row(dev_inp, out=dev_out1)
        dev_out2 = sum_row(dev_inp)
        dev_out1.copy_to_host(out1)
        dev_out2.copy_to_host(out2)
        for i in range(inp.shape[0]):
            self.assertTrue(out1[i] == inp[i].sum())
            self.assertTrue(out2[i] == inp[i].sum())

    def test_gufunc_scalar_output_bug(self):
        if False:
            i = 10
            return i + 15

        @guvectorize(['void(int32, int32[:])'], '()->()', target='cuda')
        def twice(inp, out):
            if False:
                return 10
            out[0] = inp * 2
        self.assertEqual(twice(10), 20)
        arg = np.arange(10).astype(np.int32)
        self.assertPreciseEqual(twice(arg), arg * 2)

    def test_gufunc_scalar_input_saxpy(self):
        if False:
            return 10

        @guvectorize(['void(float32, float32[:], float32[:], float32[:])'], '(),(t),(t)->(t)', target='cuda')
        def saxpy(a, x, y, out):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(out.shape[0]):
                out[i] = a * x[i] + y[i]
        A = np.float32(2)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)
        for j in range(5):
            for i in range(2):
                exp = A * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i])
        X = np.arange(10, dtype=np.float32)
        Y = np.arange(10, dtype=np.float32)
        out = saxpy(A, X, Y)
        for j in range(10):
            exp = A * X[j] + Y[j]
            self.assertTrue(exp == out[j], (exp, out[j]))
        A = np.arange(5, dtype=np.float32)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)
        for j in range(5):
            for i in range(2):
                exp = A[j] * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i], (exp, out[j, i]))

    def test_gufunc_scalar_cast(self):
        if False:
            print('Hello World!')

        @guvectorize(['void(int32, int32[:], int32[:])'], '(),(t)->(t)', target='cuda')
        def foo(a, b, out):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(b.size):
                out[i] = a * b[i]
        a = np.int64(2)
        b = np.arange(10).astype(np.int32)
        out = foo(a, b)
        np.testing.assert_equal(out, a * b)
        a = np.array(a)
        da = cuda.to_device(a)
        self.assertEqual(da.dtype, np.int64)
        with self.assertRaises(TypeError) as raises:
            foo(da, b)
        self.assertIn('does not support .astype()', str(raises.exception))

    def test_gufunc_old_style_scalar_as_array(self):
        if False:
            for i in range(10):
                print('nop')

        @guvectorize(['void(int32[:],int32[:],int32[:])'], '(n),()->(n)', target='cuda')
        def gufunc(x, y, res):
            if False:
                return 10
            for i in range(x.shape[0]):
                res[i] = x[i] + y[0]
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        b = np.array([2], dtype=np.int32)
        res = np.zeros(4, dtype=np.int32)
        expected = res.copy()
        expected = a + b
        gufunc(a, b, out=res)
        np.testing.assert_almost_equal(expected, res)
        a = np.array([1, 2, 3, 4] * 2, dtype=np.int32).reshape(2, 4)
        b = np.array([2, 10], dtype=np.int32)
        res = np.zeros((2, 4), dtype=np.int32)
        expected = res.copy()
        expected[0] = a[0] + b[0]
        expected[1] = a[1] + b[1]
        gufunc(a, b, res)
        np.testing.assert_almost_equal(expected, res)
if __name__ == '__main__':
    unittest.main()