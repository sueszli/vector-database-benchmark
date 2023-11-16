import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch

@skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
class TestCudaArrayInterface(ContextResettingTestCase):

    def assertPointersEqual(self, a, b):
        if False:
            while True:
                i = 10
        if driver.USE_NV_BINDING:
            self.assertEqual(int(a.device_ctypes_pointer), int(b.device_ctypes_pointer))

    def test_as_cuda_array(self):
        if False:
            while True:
                i = 10
        h_arr = np.arange(10)
        self.assertFalse(cuda.is_cuda_array(h_arr))
        d_arr = cuda.to_device(h_arr)
        self.assertTrue(cuda.is_cuda_array(d_arr))
        my_arr = ForeignArray(d_arr)
        self.assertTrue(cuda.is_cuda_array(my_arr))
        wrapped = cuda.as_cuda_array(my_arr)
        self.assertTrue(cuda.is_cuda_array(wrapped))
        np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr)
        np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr)
        self.assertPointersEqual(wrapped, d_arr)

    def get_stream_value(self, stream):
        if False:
            return 10
        if driver.USE_NV_BINDING:
            return int(stream.handle)
        else:
            return stream.handle.value

    @skip_if_external_memmgr('Ownership not relevant with external memmgr')
    def test_ownership(self):
        if False:
            while True:
                i = 10
        ctx = cuda.current_context()
        deallocs = ctx.memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        d_arr = cuda.to_device(np.arange(100))
        cvted = cuda.as_cuda_array(d_arr)
        del d_arr
        self.assertEqual(len(deallocs), 0)
        np.testing.assert_equal(cvted.copy_to_host(), np.arange(100))
        del cvted
        self.assertEqual(len(deallocs), 1)
        deallocs.clear()

    def test_kernel_arg(self):
        if False:
            return 10
        h_arr = np.arange(10)
        d_arr = cuda.to_device(h_arr)
        my_arr = ForeignArray(d_arr)
        wrapped = cuda.as_cuda_array(my_arr)

        @cuda.jit
        def mutate(arr, val):
            if False:
                print('Hello World!')
            i = cuda.grid(1)
            if i >= len(arr):
                return
            arr[i] += val
        val = 7
        mutate.forall(wrapped.size)(wrapped, val)
        np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr + val)
        np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr + val)

    def test_ufunc_arg(self):
        if False:
            for i in range(10):
                print('nop')

        @vectorize(['f8(f8, f8)'], target='cuda')
        def vadd(a, b):
            if False:
                print('Hello World!')
            return a + b
        h_arr = np.random.random(10)
        arr = ForeignArray(cuda.to_device(h_arr))
        val = 6
        out = vadd(arr, val)
        np.testing.assert_array_equal(out.copy_to_host(), h_arr + val)
        out = ForeignArray(cuda.device_array(h_arr.shape))
        returned = vadd(h_arr, val, out=out)
        np.testing.assert_array_equal(returned.copy_to_host(), h_arr + val)

    def test_gufunc_arg(self):
        if False:
            return 10

        @guvectorize(['(f8, f8, f8[:])'], '(),()->()', target='cuda')
        def vadd(inp, val, out):
            if False:
                print('Hello World!')
            out[0] = inp + val
        h_arr = np.random.random(10)
        arr = ForeignArray(cuda.to_device(h_arr))
        val = np.float64(7)
        out = vadd(arr, val)
        np.testing.assert_array_equal(out.copy_to_host(), h_arr + val)
        out = ForeignArray(cuda.device_array(h_arr.shape))
        returned = vadd(h_arr, val, out=out)
        np.testing.assert_array_equal(returned.copy_to_host(), h_arr + val)
        self.assertPointersEqual(returned, out._arr)

    def test_array_views(self):
        if False:
            for i in range(10):
                print('nop')
        'Views created via array interface support:\n            - Strided slices\n            - Strided slices\n        '
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)
        arr = cuda.as_cuda_array(c_arr)
        np.testing.assert_array_equal(arr.copy_to_host(), h_arr)
        np.testing.assert_array_equal(arr[:].copy_to_host(), h_arr)
        np.testing.assert_array_equal(arr[:5].copy_to_host(), h_arr[:5])
        np.testing.assert_array_equal(arr[::2].copy_to_host(), h_arr[::2])
        arr_strided = cuda.as_cuda_array(c_arr[::2])
        np.testing.assert_array_equal(arr_strided.copy_to_host(), h_arr[::2])
        self.assertEqual(arr[::2].shape, arr_strided.shape)
        self.assertEqual(arr[::2].strides, arr_strided.strides)
        self.assertEqual(arr[::2].dtype.itemsize, arr_strided.dtype.itemsize)
        self.assertEqual(arr[::2].alloc_size, arr_strided.alloc_size)
        self.assertEqual(arr[::2].nbytes, arr_strided.size * arr_strided.dtype.itemsize)
        arr[:5] = np.pi
        np.testing.assert_array_equal(c_arr.copy_to_host(), np.concatenate((np.full(5, np.pi), h_arr[5:])))
        arr[:5] = arr[5:]
        np.testing.assert_array_equal(c_arr.copy_to_host(), np.concatenate((h_arr[5:], h_arr[5:])))
        arr[:] = cuda.to_device(h_arr)
        np.testing.assert_array_equal(c_arr.copy_to_host(), h_arr)
        arr[::2] = np.pi
        np.testing.assert_array_equal(c_arr.copy_to_host()[::2], np.full(5, np.pi))
        np.testing.assert_array_equal(c_arr.copy_to_host()[1::2], h_arr[1::2])

    def test_negative_strided_issue(self):
        if False:
            return 10
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)

        def base_offset(orig, sliced):
            if False:
                print('Hello World!')
            return sliced['data'][0] - orig['data'][0]
        h_ai = h_arr.__array_interface__
        c_ai = c_arr.__cuda_array_interface__
        h_ai_sliced = h_arr[::-1].__array_interface__
        c_ai_sliced = c_arr[::-1].__cuda_array_interface__
        self.assertEqual(base_offset(h_ai, h_ai_sliced), base_offset(c_ai, c_ai_sliced))
        self.assertEqual(h_ai_sliced['shape'], c_ai_sliced['shape'])
        self.assertEqual(h_ai_sliced['strides'], c_ai_sliced['strides'])

    def test_negative_strided_copy_to_host(self):
        if False:
            while True:
                i = 10
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)
        sliced = c_arr[::-1]
        with self.assertRaises(NotImplementedError) as raises:
            sliced.copy_to_host()
        expected_msg = 'D->H copy not implemented for negative strides'
        self.assertIn(expected_msg, str(raises.exception))

    def test_masked_array(self):
        if False:
            while True:
                i = 10
        h_arr = np.random.random(10)
        h_mask = np.random.randint(2, size=10, dtype='bool')
        c_arr = cuda.to_device(h_arr)
        c_mask = cuda.to_device(h_mask)
        masked_cuda_array_interface = c_arr.__cuda_array_interface__.copy()
        masked_cuda_array_interface['mask'] = c_mask
        with self.assertRaises(NotImplementedError) as raises:
            cuda.from_cuda_array_interface(masked_cuda_array_interface)
        expected_msg = 'Masked arrays are not supported'
        self.assertIn(expected_msg, str(raises.exception))

    def test_zero_size_array(self):
        if False:
            for i in range(10):
                print('nop')
        c_arr = cuda.device_array(0)
        self.assertEqual(c_arr.__cuda_array_interface__['data'][0], 0)

        @cuda.jit
        def add_one(arr):
            if False:
                print('Hello World!')
            x = cuda.grid(1)
            N = arr.shape[0]
            if x < N:
                arr[x] += 1
        d_arr = ForeignArray(c_arr)
        add_one[1, 10](d_arr)

    def test_strides(self):
        if False:
            return 10
        c_arr = cuda.device_array((2, 3, 4))
        self.assertEqual(c_arr.__cuda_array_interface__['strides'], None)
        c_arr = c_arr[:, 1, :]
        self.assertNotEqual(c_arr.__cuda_array_interface__['strides'], None)

    def test_consuming_strides(self):
        if False:
            while True:
                i = 10
        hostarray = np.arange(10).reshape(2, 5)
        devarray = cuda.to_device(hostarray)
        face = devarray.__cuda_array_interface__
        self.assertIsNone(face['strides'])
        got = cuda.from_cuda_array_interface(face).copy_to_host()
        np.testing.assert_array_equal(got, hostarray)
        self.assertTrue(got.flags['C_CONTIGUOUS'])
        face['strides'] = hostarray.strides
        self.assertIsNotNone(face['strides'])
        got = cuda.from_cuda_array_interface(face).copy_to_host()
        np.testing.assert_array_equal(got, hostarray)
        self.assertTrue(got.flags['C_CONTIGUOUS'])

    def test_produce_no_stream(self):
        if False:
            i = 10
            return i + 15
        c_arr = cuda.device_array(10)
        self.assertIsNone(c_arr.__cuda_array_interface__['stream'])
        mapped_arr = cuda.mapped_array(10)
        self.assertIsNone(mapped_arr.__cuda_array_interface__['stream'])

    @linux_only
    def test_produce_managed_no_stream(self):
        if False:
            return 10
        managed_arr = cuda.managed_array(10)
        self.assertIsNone(managed_arr.__cuda_array_interface__['stream'])

    def test_produce_stream(self):
        if False:
            return 10
        s = cuda.stream()
        c_arr = cuda.device_array(10, stream=s)
        cai_stream = c_arr.__cuda_array_interface__['stream']
        stream_value = self.get_stream_value(s)
        self.assertEqual(stream_value, cai_stream)
        s = cuda.stream()
        mapped_arr = cuda.mapped_array(10, stream=s)
        cai_stream = mapped_arr.__cuda_array_interface__['stream']
        stream_value = self.get_stream_value(s)
        self.assertEqual(stream_value, cai_stream)

    @linux_only
    def test_produce_managed_stream(self):
        if False:
            print('Hello World!')
        s = cuda.stream()
        managed_arr = cuda.managed_array(10, stream=s)
        cai_stream = managed_arr.__cuda_array_interface__['stream']
        stream_value = self.get_stream_value(s)
        self.assertEqual(stream_value, cai_stream)

    def test_consume_no_stream(self):
        if False:
            i = 10
            return i + 15
        f_arr = ForeignArray(cuda.device_array(10))
        c_arr = cuda.as_cuda_array(f_arr)
        self.assertEqual(c_arr.stream, 0)

    def test_consume_stream(self):
        if False:
            i = 10
            return i + 15
        s = cuda.stream()
        f_arr = ForeignArray(cuda.device_array(10, stream=s))
        c_arr = cuda.as_cuda_array(f_arr)
        self.assertTrue(c_arr.stream.external)
        stream_value = self.get_stream_value(s)
        imported_stream_value = self.get_stream_value(c_arr.stream)
        self.assertEqual(stream_value, imported_stream_value)

    def test_consume_no_sync(self):
        if False:
            return 10
        f_arr = ForeignArray(cuda.device_array(10))
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            cuda.as_cuda_array(f_arr)
        mock_sync.assert_not_called()

    def test_consume_sync(self):
        if False:
            return 10
        s = cuda.stream()
        f_arr = ForeignArray(cuda.device_array(10, stream=s))
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            cuda.as_cuda_array(f_arr)
        mock_sync.assert_called_once_with()

    def test_consume_sync_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        s = cuda.stream()
        f_arr = ForeignArray(cuda.device_array(10, stream=s))
        with override_config('CUDA_ARRAY_INTERFACE_SYNC', False):
            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
                cuda.as_cuda_array(f_arr)
            mock_sync.assert_not_called()

    def test_launch_no_sync(self):
        if False:
            return 10
        f_arr = ForeignArray(cuda.device_array(10))

        @cuda.jit
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            pass
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            f[1, 1](f_arr)
        mock_sync.assert_not_called()

    def test_launch_sync(self):
        if False:
            i = 10
            return i + 15
        s = cuda.stream()
        f_arr = ForeignArray(cuda.device_array(10, stream=s))

        @cuda.jit
        def f(x):
            if False:
                print('Hello World!')
            pass
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            f[1, 1](f_arr)
        mock_sync.assert_called_once_with()

    def test_launch_sync_two_streams(self):
        if False:
            i = 10
            return i + 15
        s1 = cuda.stream()
        s2 = cuda.stream()
        f_arr1 = ForeignArray(cuda.device_array(10, stream=s1))
        f_arr2 = ForeignArray(cuda.device_array(10, stream=s2))

        @cuda.jit
        def f(x, y):
            if False:
                return 10
            pass
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            f[1, 1](f_arr1, f_arr2)
        mock_sync.assert_has_calls([call(), call()])

    def test_launch_sync_disabled(self):
        if False:
            print('Hello World!')
        s1 = cuda.stream()
        s2 = cuda.stream()
        f_arr1 = ForeignArray(cuda.device_array(10, stream=s1))
        f_arr2 = ForeignArray(cuda.device_array(10, stream=s2))
        with override_config('CUDA_ARRAY_INTERFACE_SYNC', False):

            @cuda.jit
            def f(x, y):
                if False:
                    return 10
                pass
            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
                f[1, 1](f_arr1, f_arr2)
            mock_sync.assert_not_called()
if __name__ == '__main__':
    unittest.main()