import unittest
import cupy
from cupy import testing

class TestCArray(unittest.TestCase):

    def test_size(self):
        if False:
            while True:
                i = 10
        x = cupy.arange(3).astype('i')
        y = cupy.ElementwiseKernel('raw int32 x', 'int32 y', 'y = x.size()', 'test_carray_size')(x, size=1)
        assert int(y[0]) == 3

    def test_shape(self):
        if False:
            print('Hello World!')
        x = cupy.arange(6).reshape((2, 3)).astype('i')
        y = cupy.ElementwiseKernel('raw int32 x', 'int32 y', 'y = x.shape()[i]', 'test_carray_shape')(x, size=2)
        testing.assert_array_equal(y, (2, 3))

    def test_strides(self):
        if False:
            i = 10
            return i + 15
        x = cupy.arange(6).reshape((2, 3)).astype('i')
        y = cupy.ElementwiseKernel('raw int32 x', 'int32 y', 'y = x.strides()[i]', 'test_carray_strides')(x, size=2)
        testing.assert_array_equal(y, (12, 4))

    def test_getitem_int(self):
        if False:
            print('Hello World!')
        x = cupy.arange(24).reshape((2, 3, 4)).astype('i')
        y = cupy.empty_like(x)
        y = cupy.ElementwiseKernel('raw T x', 'int32 y', 'y = x[i]', 'test_carray_getitem_int')(x, y)
        testing.assert_array_equal(y, x)

    def test_getitem_idx(self):
        if False:
            for i in range(10):
                print('nop')
        x = cupy.arange(24).reshape((2, 3, 4)).astype('i')
        y = cupy.empty_like(x)
        y = cupy.ElementwiseKernel('raw T x', 'int32 y', 'ptrdiff_t idx[] = {i / 12, i / 4 % 3, i % 4}; y = x[idx]', 'test_carray_getitem_idx')(x, y)
        testing.assert_array_equal(y, x)

@testing.parameterize({'size': 2 ** 31 - 1024}, {'size': 2 ** 31}, {'size': 2 ** 31 + 1024}, {'size': 2 ** 32 - 1024}, {'size': 2 ** 32}, {'size': 2 ** 32 + 1024})
@testing.slow
class TestCArray32BitBoundary(unittest.TestCase):

    def tearDown(self):
        if False:
            while True:
                i = 10
        cupy.get_default_memory_pool().free_all_blocks()

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP does not support this')
    def test(self):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.full((1, self.size), 7, dtype=cupy.int8)
        result = a.sum(axis=0, dtype=cupy.int8)
        assert result.sum(dtype=cupy.int64) == self.size * 7

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP does not support this')
    def test_assign(self):
        if False:
            i = 10
            return i + 15
        a = cupy.zeros(self.size, dtype=cupy.int8)
        a[-1] = 1.0
        assert a.sum() == 1