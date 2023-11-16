from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
recordwith2darray = np.dtype([('i', np.int32), ('j', np.float32, (3, 2))])

class TestSharedMemoryIssue(CUDATestCase):

    def test_issue_953_sm_linkage_conflict(self):
        if False:
            return 10

        @cuda.jit(device=True)
        def inner():
            if False:
                i = 10
                return i + 15
            inner_arr = cuda.shared.array(1, dtype=int32)

        @cuda.jit
        def outer():
            if False:
                print('Hello World!')
            outer_arr = cuda.shared.array(1, dtype=int32)
            inner()
        outer[1, 1]()

    def _check_shared_array_size(self, shape, expected):
        if False:
            return 10

        @cuda.jit
        def s(a):
            if False:
                i = 10
                return i + 15
            arr = cuda.shared.array(shape, dtype=int32)
            a[0] = arr.size
        result = np.zeros(1, dtype=np.int32)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_1051_shared_size_broken_1d(self):
        if False:
            print('Hello World!')
        self._check_shared_array_size(2, 2)

    def test_issue_1051_shared_size_broken_2d(self):
        if False:
            return 10
        self._check_shared_array_size((2, 3), 6)

    def test_issue_1051_shared_size_broken_3d(self):
        if False:
            return 10
        self._check_shared_array_size((2, 3, 4), 24)

    def _check_shared_array_size_fp16(self, shape, expected, ty):
        if False:
            print('Hello World!')

        @cuda.jit
        def s(a):
            if False:
                return 10
            arr = cuda.shared.array(shape, dtype=ty)
            a[0] = arr.size
        result = np.zeros(1, dtype=np.float16)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_fp16_support(self):
        if False:
            print('Hello World!')
        self._check_shared_array_size_fp16(2, 2, types.float16)
        self._check_shared_array_size_fp16(2, 2, np.float16)

    def test_issue_2393(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test issue of warp misalign address due to nvvm not knowing the\n        alignment(? but it should have taken the natural alignment of the type)\n        '
        num_weights = 2
        num_blocks = 48
        examples_per_block = 4
        threads_per_block = 1

        @cuda.jit
        def costs_func(d_block_costs):
            if False:
                i = 10
                return i + 15
            s_features = cuda.shared.array((examples_per_block, num_weights), float64)
            s_initialcost = cuda.shared.array(7, float64)
            threadIdx = cuda.threadIdx.x
            prediction = 0
            for j in range(num_weights):
                prediction += s_features[threadIdx, j]
            d_block_costs[0] = s_initialcost[0] + prediction
        block_costs = np.zeros(num_blocks, dtype=np.float64)
        d_block_costs = cuda.to_device(block_costs)
        costs_func[num_blocks, threads_per_block](d_block_costs)
        cuda.synchronize()

class TestSharedMemory(CUDATestCase):

    def _test_shared(self, arr):
        if False:
            i = 10
            return i + 15
        nelem = len(arr)
        nthreads = 16
        nblocks = int(nelem / nthreads)
        dt = nps.from_dtype(arr.dtype)

        @cuda.jit
        def use_sm_chunk_copy(x, y):
            if False:
                while True:
                    i = 10
            sm = cuda.shared.array(nthreads, dtype=dt)
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            bd = cuda.blockDim.x
            i = bx * bd + tx
            if i < len(x):
                sm[tx] = x[i]
            cuda.syncthreads()
            if tx == 0:
                for j in range(nthreads):
                    y[bd * bx + j] = sm[j]
        d_result = cuda.device_array_like(arr)
        use_sm_chunk_copy[nblocks, nthreads](arr, d_result)
        host_result = d_result.copy_to_host()
        np.testing.assert_array_equal(arr, host_result)

    def test_shared_recarray(self):
        if False:
            while True:
                i = 10
        arr = np.recarray(128, dtype=recordwith2darray)
        for x in range(len(arr)):
            arr[x].i = x
            j = np.arange(3 * 2, dtype=np.float32)
            arr[x].j = j.reshape(3, 2) * x
        self._test_shared(arr)

    def test_shared_bool(self):
        if False:
            return 10
        arr = np.random.randint(2, size=(1024,), dtype=np.bool_)
        self._test_shared(arr)

    def _test_dynshared_slice(self, func, arr, expected):
        if False:
            for i in range(10):
                print('nop')
        nshared = arr.size * arr.dtype.itemsize
        func[1, 1, 0, nshared](arr)
        np.testing.assert_array_equal(expected, arr)

    def test_dynshared_slice_write(self):
        if False:
            while True:
                i = 10

        @cuda.jit
        def slice_write(x):
            if False:
                print('Hello World!')
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:2]
            sm1[0] = 1
            sm2[0] = 2
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
        arr = np.zeros(2, dtype=np.int32)
        expected = np.array([1, 2], dtype=np.int32)
        self._test_dynshared_slice(slice_write, arr, expected)

    def test_dynshared_slice_read(self):
        if False:
            print('Hello World!')

        @cuda.jit
        def slice_read(x):
            if False:
                for i in range(10):
                    print('nop')
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:2]
            dynsmem[0] = 1
            dynsmem[1] = 2
            x[0] = sm1[0]
            x[1] = sm2[0]
        arr = np.zeros(2, dtype=np.int32)
        expected = np.array([1, 2], dtype=np.int32)
        self._test_dynshared_slice(slice_read, arr, expected)

    def test_dynshared_slice_diff_sizes(self):
        if False:
            while True:
                i = 10

        @cuda.jit
        def slice_diff_sizes(x):
            if False:
                for i in range(10):
                    print('nop')
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:3]
            dynsmem[0] = 1
            dynsmem[1] = 2
            dynsmem[2] = 3
            x[0] = sm1[0]
            x[1] = sm2[0]
            x[2] = sm2[1]
        arr = np.zeros(3, dtype=np.int32)
        expected = np.array([1, 2, 3], dtype=np.int32)
        self._test_dynshared_slice(slice_diff_sizes, arr, expected)

    def test_dynshared_slice_overlap(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit
        def slice_overlap(x):
            if False:
                while True:
                    i = 10
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:2]
            sm2 = dynsmem[1:4]
            dynsmem[0] = 1
            dynsmem[1] = 2
            dynsmem[2] = 3
            dynsmem[3] = 4
            x[0] = sm1[0]
            x[1] = sm1[1]
            x[2] = sm2[0]
            x[3] = sm2[1]
            x[4] = sm2[2]
        arr = np.zeros(5, dtype=np.int32)
        expected = np.array([1, 2, 2, 3, 4], dtype=np.int32)
        self._test_dynshared_slice(slice_overlap, arr, expected)

    def test_dynshared_slice_gaps(self):
        if False:
            return 10

        @cuda.jit
        def slice_gaps(x):
            if False:
                print('Hello World!')
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[1:3]
            sm2 = dynsmem[4:6]
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99
            dynsmem[6] = 99
            sm1[0] = 1
            sm1[1] = 2
            sm2[0] = 3
            sm2[1] = 4
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]
            x[6] = dynsmem[6]
        arr = np.zeros(7, dtype=np.int32)
        expected = np.array([99, 1, 2, 99, 3, 4, 99], dtype=np.int32)
        self._test_dynshared_slice(slice_gaps, arr, expected)

    def test_dynshared_slice_write_backwards(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit
        def slice_write_backwards(x):
            if False:
                return 10
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[1::-1]
            sm2 = dynsmem[3:1:-1]
            sm1[0] = 1
            sm1[1] = 2
            sm2[0] = 3
            sm2[1] = 4
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
        arr = np.zeros(4, dtype=np.int32)
        expected = np.array([2, 1, 4, 3], dtype=np.int32)
        self._test_dynshared_slice(slice_write_backwards, arr, expected)

    def test_dynshared_slice_nonunit_stride(self):
        if False:
            print('Hello World!')

        @cuda.jit
        def slice_nonunit_stride(x):
            if False:
                i = 10
                return i + 15
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[::2]
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99
            sm1[0] = 1
            sm1[1] = 2
            sm1[2] = 3
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]
        arr = np.zeros(6, dtype=np.int32)
        expected = np.array([1, 99, 2, 99, 3, 99], dtype=np.int32)
        self._test_dynshared_slice(slice_nonunit_stride, arr, expected)

    def test_dynshared_slice_nonunit_reverse_stride(self):
        if False:
            print('Hello World!')

        @cuda.jit
        def slice_nonunit_reverse_stride(x):
            if False:
                return 10
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[-1::-2]
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99
            sm1[0] = 1
            sm1[1] = 2
            sm1[2] = 3
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]
        arr = np.zeros(6, dtype=np.int32)
        expected = np.array([99, 3, 99, 2, 99, 1], dtype=np.int32)
        self._test_dynshared_slice(slice_nonunit_reverse_stride, arr, expected)

    def test_issue_5073(self):
        if False:
            print('Hello World!')
        arr = np.arange(1024)
        nelem = len(arr)
        nthreads = 16
        nblocks = int(nelem / nthreads)
        dt = nps.from_dtype(arr.dtype)
        nshared = nthreads * arr.dtype.itemsize
        chunksize = int(nthreads / 2)

        @cuda.jit
        def sm_slice_copy(x, y, chunksize):
            if False:
                return 10
            dynsmem = cuda.shared.array(0, dtype=dt)
            sm1 = dynsmem[0:chunksize]
            sm2 = dynsmem[chunksize:chunksize * 2]
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            bd = cuda.blockDim.x
            i = bx * bd + tx
            if i < len(x):
                if tx < chunksize:
                    sm1[tx] = x[i]
                else:
                    sm2[tx - chunksize] = x[i]
            cuda.syncthreads()
            if tx == 0:
                for j in range(chunksize):
                    y[bd * bx + j] = sm1[j]
                    y[bd * bx + j + chunksize] = sm2[j]
        d_result = cuda.device_array_like(arr)
        sm_slice_copy[nblocks, nthreads, 0, nshared](arr, d_result, chunksize)
        host_result = d_result.copy_to_host()
        np.testing.assert_array_equal(arr, host_result)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_invalid_array_type(self):
        if False:
            for i in range(10):
                print('nop')
        rgx = ".*Cannot infer the type of variable 'arr'.*"

        def unsupported_type():
            if False:
                i = 10
                return i + 15
            arr = cuda.shared.array(10, dtype=np.dtype('O'))
        with self.assertRaisesRegex(TypingError, rgx):
            cuda.jit(void())(unsupported_type)
        rgx = ".*Invalid NumPy dtype specified: 'int33'.*"

        def invalid_string_type():
            if False:
                return 10
            arr = cuda.shared.array(10, dtype='int33')
        with self.assertRaisesRegex(TypingError, rgx):
            cuda.jit(void())(invalid_string_type)

    @skip_on_cudasim('Struct model array unsupported in simulator')
    def test_struct_model_type_static(self):
        if False:
            return 10
        nthreads = 64

        @cuda.jit(void(int32[::1], int32[::1]))
        def write_then_reverse_read_static(outx, outy):
            if False:
                i = 10
                return i + 15
            arr = cuda.shared.array(nthreads, dtype=test_struct_model_type)
            i = cuda.grid(1)
            ri = nthreads - i - 1
            if i < len(outx) and i < len(outy):
                obj = TestStruct(int32(i), int32(i * 2))
                arr[i] = obj
                cuda.syncthreads()
                outx[i] = arr[ri].x
                outy[i] = arr[ri].y
        arrx = np.zeros((nthreads,), dtype='int32')
        arry = np.zeros((nthreads,), dtype='int32')
        write_then_reverse_read_static[1, nthreads](arrx, arry)
        for (i, x) in enumerate(arrx):
            self.assertEqual(x, nthreads - i - 1)
        for (i, y) in enumerate(arry):
            self.assertEqual(y, (nthreads - i - 1) * 2)
if __name__ == '__main__':
    unittest.main()