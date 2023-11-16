import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config

def useful_syncwarp(ary):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    if i == 0:
        ary[0] = 42
    cuda.syncwarp(4294967295)
    ary[i] = ary[0]

def use_shfl_sync_idx(ary, idx):
    if False:
        for i in range(10):
            print('nop')
    i = cuda.grid(1)
    val = cuda.shfl_sync(4294967295, i, idx)
    ary[i] = val

def use_shfl_sync_up(ary, delta):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    val = cuda.shfl_up_sync(4294967295, i, delta)
    ary[i] = val

def use_shfl_sync_down(ary, delta):
    if False:
        while True:
            i = 10
    i = cuda.grid(1)
    val = cuda.shfl_down_sync(4294967295, i, delta)
    ary[i] = val

def use_shfl_sync_xor(ary, xor):
    if False:
        while True:
            i = 10
    i = cuda.grid(1)
    val = cuda.shfl_xor_sync(4294967295, i, xor)
    ary[i] = val

def use_shfl_sync_with_val(ary, into):
    if False:
        i = 10
        return i + 15
    i = cuda.grid(1)
    val = cuda.shfl_sync(4294967295, into, 0)
    ary[i] = val

def use_vote_sync_all(ary_in, ary_out):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    pred = cuda.all_sync(4294967295, ary_in[i])
    ary_out[i] = pred

def use_vote_sync_any(ary_in, ary_out):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    pred = cuda.any_sync(4294967295, ary_in[i])
    ary_out[i] = pred

def use_vote_sync_eq(ary_in, ary_out):
    if False:
        print('Hello World!')
    i = cuda.grid(1)
    pred = cuda.eq_sync(4294967295, ary_in[i])
    ary_out[i] = pred

def use_vote_sync_ballot(ary):
    if False:
        for i in range(10):
            print('nop')
    i = cuda.threadIdx.x
    ballot = cuda.ballot_sync(4294967295, True)
    ary[i] = ballot

def use_match_any_sync(ary_in, ary_out):
    if False:
        return 10
    i = cuda.grid(1)
    ballot = cuda.match_any_sync(4294967295, ary_in[i])
    ary_out[i] = ballot

def use_match_all_sync(ary_in, ary_out):
    if False:
        while True:
            i = 10
    i = cuda.grid(1)
    (ballot, pred) = cuda.match_all_sync(4294967295, ary_in[i])
    ary_out[i] = ballot if pred else 0

def use_independent_scheduling(arr):
    if False:
        while True:
            i = 10
    i = cuda.threadIdx.x
    if i % 4 == 0:
        ballot = cuda.ballot_sync(286331153, True)
    elif i % 4 == 1:
        ballot = cuda.ballot_sync(572662306, True)
    elif i % 4 == 2:
        ballot = cuda.ballot_sync(1145324612, True)
    elif i % 4 == 3:
        ballot = cuda.ballot_sync(2290649224, True)
    arr[i] = ballot

def _safe_cc_check(cc):
    if False:
        i = 10
        return i + 15
    if config.ENABLE_CUDASIM:
        return True
    else:
        return cuda.get_current_device().compute_capability >= cc

@skip_on_cudasim('Warp Operations are not yet implemented on cudasim')
class TestCudaWarpOperations(CUDATestCase):

    def test_useful_syncwarp(self):
        if False:
            return 10
        compiled = cuda.jit('void(int32[:])')(useful_syncwarp)
        nelem = 32
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == 42))

    def test_shfl_sync_idx(self):
        if False:
            print('Hello World!')
        compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_idx)
        nelem = 32
        idx = 4
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary, idx)
        self.assertTrue(np.all(ary == idx))

    def test_shfl_sync_up(self):
        if False:
            while True:
                i = 10
        compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_up)
        nelem = 32
        delta = 4
        ary = np.empty(nelem, dtype=np.int32)
        exp = np.arange(nelem, dtype=np.int32)
        exp[delta:] -= delta
        compiled[1, nelem](ary, delta)
        self.assertTrue(np.all(ary == exp))

    def test_shfl_sync_down(self):
        if False:
            i = 10
            return i + 15
        compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_down)
        nelem = 32
        delta = 4
        ary = np.empty(nelem, dtype=np.int32)
        exp = np.arange(nelem, dtype=np.int32)
        exp[:-delta] += delta
        compiled[1, nelem](ary, delta)
        self.assertTrue(np.all(ary == exp))

    def test_shfl_sync_xor(self):
        if False:
            while True:
                i = 10
        compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_xor)
        nelem = 32
        xor = 16
        ary = np.empty(nelem, dtype=np.int32)
        exp = np.arange(nelem, dtype=np.int32) ^ xor
        compiled[1, nelem](ary, xor)
        self.assertTrue(np.all(ary == exp))

    def test_shfl_sync_types(self):
        if False:
            print('Hello World!')
        types = (int32, int64, float32, float64)
        values = (np.int32(-1), np.int64(1 << 42), np.float32(np.pi), np.float64(np.pi))
        for (typ, val) in zip(types, values):
            compiled = cuda.jit((typ[:], typ))(use_shfl_sync_with_val)
            nelem = 32
            ary = np.empty(nelem, dtype=val.dtype)
            compiled[1, nelem](ary, val)
            self.assertTrue(np.all(ary == val))

    def test_vote_sync_all(self):
        if False:
            while True:
                i = 10
        compiled = cuda.jit('void(int32[:], int32[:])')(use_vote_sync_all)
        nelem = 32
        ary_in = np.ones(nelem, dtype=np.int32)
        ary_out = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))
        ary_in[-1] = 0
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))

    def test_vote_sync_any(self):
        if False:
            while True:
                i = 10
        compiled = cuda.jit('void(int32[:], int32[:])')(use_vote_sync_any)
        nelem = 32
        ary_in = np.zeros(nelem, dtype=np.int32)
        ary_out = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))
        ary_in[2] = 1
        ary_in[5] = 1
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))

    def test_vote_sync_eq(self):
        if False:
            print('Hello World!')
        compiled = cuda.jit('void(int32[:], int32[:])')(use_vote_sync_eq)
        nelem = 32
        ary_in = np.zeros(nelem, dtype=np.int32)
        ary_out = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))
        ary_in[1] = 1
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))
        ary_in[:] = 1
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))

    def test_vote_sync_ballot(self):
        if False:
            print('Hello World!')
        compiled = cuda.jit('void(uint32[:])')(use_vote_sync_ballot)
        nelem = 32
        ary = np.empty(nelem, dtype=np.uint32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == np.uint32(4294967295)))

    @unittest.skipUnless(_safe_cc_check((7, 0)), 'Matching requires at least Volta Architecture')
    def test_match_any_sync(self):
        if False:
            i = 10
            return i + 15
        compiled = cuda.jit('void(int32[:], int32[:])')(use_match_any_sync)
        nelem = 10
        ary_in = np.arange(nelem, dtype=np.int32) % 2
        ary_out = np.empty(nelem, dtype=np.int32)
        exp = np.tile((341, 682), 5)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == exp))

    @unittest.skipUnless(_safe_cc_check((7, 0)), 'Matching requires at least Volta Architecture')
    def test_match_all_sync(self):
        if False:
            i = 10
            return i + 15
        compiled = cuda.jit('void(int32[:], int32[:])')(use_match_all_sync)
        nelem = 10
        ary_in = np.zeros(nelem, dtype=np.int32)
        ary_out = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1023))
        ary_in[1] = 4
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))

    @unittest.skipUnless(_safe_cc_check((7, 0)), 'Independent scheduling requires at least Volta Architecture')
    def test_independent_scheduling(self):
        if False:
            return 10
        compiled = cuda.jit('void(uint32[:])')(use_independent_scheduling)
        arr = np.empty(32, dtype=np.uint32)
        exp = np.tile((286331153, 572662306, 1145324612, 2290649224), 8)
        compiled[1, 32](arr)
        self.assertTrue(np.all(arr == exp))

    def test_activemask(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit
        def use_activemask(x):
            if False:
                i = 10
                return i + 15
            i = cuda.grid(1)
            if i % 2 == 0:
                x[i] = cuda.activemask()
            else:
                x[i] = cuda.activemask()
        out = np.zeros(32, dtype=np.uint32)
        use_activemask[1, 32](out)
        expected = np.tile((1431655765, 2863311530), 16)
        np.testing.assert_equal(expected, out)

    def test_lanemask_lt(self):
        if False:
            print('Hello World!')

        @cuda.jit
        def use_lanemask_lt(x):
            if False:
                for i in range(10):
                    print('nop')
            i = cuda.grid(1)
            x[i] = cuda.lanemask_lt()
        out = np.zeros(32, dtype=np.uint32)
        use_lanemask_lt[1, 32](out)
        expected = np.asarray([2 ** i - 1 for i in range(32)], dtype=np.uint32)
        np.testing.assert_equal(expected, out)
if __name__ == '__main__':
    unittest.main()