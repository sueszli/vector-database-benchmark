from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm

def test_ensure_platform_int():
    if False:
        for i in range(10):
            print('nop')
    arr = np.arange(100, dtype=np.intp)
    result = libalgos.ensure_platform_int(arr)
    assert result is arr

def test_is_lexsorted():
    if False:
        print('Hello World!')
    failure = [np.array([3] * 32 + [2] * 32 + [1] * 32 + [0] * 32, dtype='int64'), np.array(list(range(31))[::-1] * 4, dtype='int64')]
    assert not libalgos.is_lexsorted(failure)

def test_groupsort_indexer():
    if False:
        for i in range(10):
            print('nop')
    a = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    b = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    result = libalgos.groupsort_indexer(a, 1000)[0]
    expected = np.argsort(a, kind='mergesort')
    expected = expected.astype(np.intp)
    tm.assert_numpy_array_equal(result, expected)
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.intp)
    tm.assert_numpy_array_equal(result, expected)

class TestPadBackfill:

    def test_backfill(self):
        if False:
            return 10
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)
        filler = libalgos.backfill['int64_t'](old, new)
        expect_filler = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)
        old = np.array([1, 4], dtype=np.int64)
        new = np.array(list(range(5, 10)), dtype=np.int64)
        filler = libalgos.backfill['int64_t'](old, new)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad(self):
        if False:
            for i in range(10):
                print('nop')
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)
        filler = libalgos.pad['int64_t'](old, new)
        expect_filler = np.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)
        old = np.array([5, 10], dtype=np.int64)
        new = np.arange(5, dtype=np.int64)
        filler = libalgos.pad['int64_t'](old, new)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad_backfill_object_segfault(self):
        if False:
            i = 10
            return i + 15
        old = np.array([], dtype='O')
        new = np.array([datetime(2010, 12, 31)], dtype='O')
        result = libalgos.pad['object'](old, new)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = libalgos.pad['object'](new, old)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = libalgos.backfill['object'](old, new)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = libalgos.backfill['object'](new, old)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

class TestInfinity:

    def test_infinity_sort(self):
        if False:
            return 10
        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()
        ref_nums = [NegInf, float('-inf'), -1e+100, 0, 1e+100, float('inf'), Inf]
        assert all((Inf >= x for x in ref_nums))
        assert all((Inf > x or x is Inf for x in ref_nums))
        assert Inf >= Inf and Inf == Inf
        assert not Inf < Inf and (not Inf > Inf)
        assert libalgos.Infinity() == libalgos.Infinity()
        assert not libalgos.Infinity() != libalgos.Infinity()
        assert all((NegInf <= x for x in ref_nums))
        assert all((NegInf < x or x is NegInf for x in ref_nums))
        assert NegInf <= NegInf and NegInf == NegInf
        assert not NegInf < NegInf and (not NegInf > NegInf)
        assert libalgos.NegInfinity() == libalgos.NegInfinity()
        assert not libalgos.NegInfinity() != libalgos.NegInfinity()
        for perm in permutations(ref_nums):
            assert sorted(perm) == ref_nums
        np.array([libalgos.Infinity()] * 32).argsort()
        np.array([libalgos.NegInfinity()] * 32).argsort()

    def test_infinity_against_nan(self):
        if False:
            while True:
                i = 10
        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()
        assert not Inf > np.nan
        assert not Inf >= np.nan
        assert not Inf < np.nan
        assert not Inf <= np.nan
        assert not Inf == np.nan
        assert Inf != np.nan
        assert not NegInf > np.nan
        assert not NegInf >= np.nan
        assert not NegInf < np.nan
        assert not NegInf <= np.nan
        assert not NegInf == np.nan
        assert NegInf != np.nan