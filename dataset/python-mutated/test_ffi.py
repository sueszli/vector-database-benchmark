import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import skip_unless_cffi

@skip_unless_cffi
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestFFI(CUDATestCase):

    def test_ex_linking_cu(self):
        if False:
            i = 10
            return i + 15
        from numba import cuda
        import numpy as np
        import os
        mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')
        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')

        @cuda.jit(link=[functions_cu])
        def multiply_vectors(r, x, y):
            if False:
                while True:
                    i = 10
            i = cuda.grid(1)
            if i < len(r):
                r[i] = mul(x[i], y[i])
        N = 32
        np.random.seed(1)
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        r = np.zeros_like(x)
        multiply_vectors[1, 32](r, x, y)
        np.testing.assert_array_equal(r, x * y)

    def test_ex_from_buffer(self):
        if False:
            i = 10
            return i + 15
        from numba import cuda
        import os
        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')
        signature = 'float32(CPointer(float32), int32)'
        sum_reduce = cuda.declare_device('sum_reduce', signature)
        import cffi
        ffi = cffi.FFI()

        @cuda.jit(link=[functions_cu])
        def reduction_caller(result, array):
            if False:
                print('Hello World!')
            array_ptr = ffi.from_buffer(array)
            result[()] = sum_reduce(array_ptr, len(array))
        import numpy as np
        x = np.arange(10).astype(np.float32)
        r = np.ndarray((), dtype=np.float32)
        reduction_caller[1, 1](r, x)
        expected = np.sum(x)
        actual = r[()]
        np.testing.assert_allclose(expected, actual)
if __name__ == '__main__':
    unittest.main()