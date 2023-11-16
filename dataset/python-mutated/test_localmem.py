import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct

def culocal(A, B):
    if False:
        i = 10
        return i + 15
    C = cuda.local.array(1000, dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]

def culocalcomplex(A, B):
    if False:
        i = 10
        return i + 15
    C = cuda.local.array(100, dtype=complex128)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]

def culocal1tuple(A, B):
    if False:
        i = 10
        return i + 15
    C = cuda.local.array((5,), dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]

@skip_on_cudasim('PTX inspection not available in cudasim')
class TestCudaLocalMem(CUDATestCase):

    def test_local_array(self):
        if False:
            print('Hello World!')
        sig = (int32[:], int32[:])
        jculocal = cuda.jit(sig)(culocal)
        self.assertTrue('.local' in jculocal.inspect_asm(sig))
        A = np.arange(1000, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_1_tuple(self):
        if False:
            while True:
                i = 10
        'Ensure that local arrays can be constructed with 1-tuple shape\n        '
        jculocal = cuda.jit('void(int32[:], int32[:])')(culocal1tuple)
        A = np.arange(5, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_complex(self):
        if False:
            return 10
        sig = 'void(complex128[:], complex128[:])'
        jculocalcomplex = cuda.jit(sig)(culocalcomplex)
        A = (np.arange(100, dtype='complex128') - 1) / 2j
        B = np.zeros_like(A)
        jculocalcomplex[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def check_dtype(self, f, dtype):
        if False:
            for i in range(10):
                print('nop')
        annotation = next(iter(f.overloads.values()))._type_annotation
        l_dtype = annotation.typemap['l'].dtype
        self.assertEqual(l_dtype, dtype)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numba_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit(void(int32[::1]))
        def f(x):
            if False:
                while True:
                    i = 10
            l = cuda.local.array(10, dtype=int32)
            l[0] = x[0]
            x[0] = l[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numpy_dtype(self):
        if False:
            i = 10
            return i + 15

        @cuda.jit(void(int32[::1]))
        def f(x):
            if False:
                while True:
                    i = 10
            l = cuda.local.array(10, dtype=np.int32)
            l[0] = x[0]
            x[0] = l[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_string_dtype(self):
        if False:
            return 10

        @cuda.jit(void(int32[::1]))
        def f(x):
            if False:
                print('Hello World!')
            l = cuda.local.array(10, dtype='int32')
            l[0] = x[0]
            x[0] = l[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_invalid_string_dtype(self):
        if False:
            return 10
        re = ".*Invalid NumPy dtype specified: 'int33'.*"
        with self.assertRaisesRegex(TypingError, re):

            @cuda.jit(void(int32[::1]))
            def f(x):
                if False:
                    print('Hello World!')
                l = cuda.local.array(10, dtype='int33')
                l[0] = x[0]
                x[0] = l[0]

    def test_type_with_struct_data_model(self):
        if False:
            print('Hello World!')

        @cuda.jit(void(test_struct_model_type[::1]))
        def f(x):
            if False:
                return 10
            l = cuda.local.array(10, dtype=test_struct_model_type)
            l[0] = x[0]
            x[0] = l[0]
        self.check_dtype(f, test_struct_model_type)

    def test_struct_model_type_arr(self):
        if False:
            print('Hello World!')

        @cuda.jit(void(int32[::1], int32[::1]))
        def f(outx, outy):
            if False:
                while True:
                    i = 10
            arr = cuda.local.array(10, dtype=test_struct_model_type)
            for i in range(len(arr)):
                obj = TestStruct(int32(i), int32(i * 2))
                arr[i] = obj
            for i in range(len(arr)):
                outx[i] = arr[i].x
                outy[i] = arr[i].y
        arrx = np.array((10,), dtype='int32')
        arry = np.array((10,), dtype='int32')
        f[1, 1](arrx, arry)
        for (i, x) in enumerate(arrx):
            self.assertEqual(x, i)
        for (i, y) in enumerate(arry):
            self.assertEqual(y, i * 2)

    def _check_local_array_size_fp16(self, shape, expected, ty):
        if False:
            print('Hello World!')

        @cuda.jit
        def s(a):
            if False:
                while True:
                    i = 10
            arr = cuda.local.array(shape, dtype=ty)
            a[0] = arr.size
        result = np.zeros(1, dtype=np.float16)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_fp16_support(self):
        if False:
            print('Hello World!')
        self._check_local_array_size_fp16(2, 2, types.float16)
        self._check_local_array_size_fp16(2, 2, np.float16)
if __name__ == '__main__':
    unittest.main()