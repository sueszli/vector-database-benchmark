from itertools import combinations
import unittest
import sys
import cupy
from cupy import _environment
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cub_reduction
from cupy.cuda import memory

@unittest.skipIf(_environment.get_cub_path() is None, 'CUB not found')
class CubReductionTestBase(unittest.TestCase):
    """
    Note: call self.can_use() when arrays are already allocated, otherwise
    call self._test_can_use().
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if cupy.cuda.runtime.is_hip:
            if _environment.get_hipcc_path() is None:
                self.skipTest('hipcc is not found')
        self.can_use = cupy._core._cub_reduction._can_use_cub_block_reduction
        self.old_accelerators = _accelerator.get_reduction_accelerators()
        _accelerator.set_reduction_accelerators(['cub'])

    def tearDown(self):
        if False:
            while True:
                i = 10
        _accelerator.set_reduction_accelerators(self.old_accelerators)

    def _test_can_use(self, i_shape, o_shape, r_axis, o_axis, order, expected):
        if False:
            i = 10
            return i + 15
        in_args = [cupy.testing.shaped_arange(i_shape, order=order)]
        out_args = [cupy.testing.shaped_arange(o_shape, order=order)]
        result = self.can_use(in_args, out_args, r_axis, o_axis) is not None
        assert result is expected

@testing.parameterize(*testing.product({'shape': [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)], 'order': ('C', 'F')}))
class TestSimpleCubReductionKernelContiguity(CubReductionTestBase):

    @testing.for_contiguous_axes()
    def test_can_use_cub_contiguous(self, axis):
        if False:
            return 10
        r_axis = axis
        i_shape = self.shape
        o_axis = tuple((i for i in range(len(i_shape)) if i not in r_axis))
        o_shape = tuple((self.shape[i] for i in o_axis))
        self._test_can_use(i_shape, o_shape, r_axis, o_axis, self.order, True)

    @testing.for_contiguous_axes()
    def test_can_use_cub_non_contiguous(self, axis):
        if False:
            for i in range(10):
                print('nop')
        dim = len(self.shape)
        r_dim = len(axis)
        non_contiguous_axes = [i for i in combinations(range(dim), r_dim) if i != axis]
        i_shape = self.shape
        for r_axis in non_contiguous_axes:
            o_axis = tuple((i for i in range(dim) if i not in r_axis))
            o_shape = tuple((self.shape[i] for i in o_axis))
            self._test_can_use(i_shape, o_shape, r_axis, o_axis, self.order, False)

class TestSimpleCubReductionKernelMisc(CubReductionTestBase):

    def test_can_use_cub_nonsense_input1(self):
        if False:
            while True:
                i = 10
        a = cupy.random.random((2, 3, 4))
        b = cupy.random.random((2, 3, 4))
        c = cupy.empty((2, 3))
        assert self.can_use([a, b], [c], (2,), (0, 1)) is None

    def test_can_use_cub_nonsense_input2(self):
        if False:
            print('Hello World!')
        self._test_can_use((2, 3, 4), (2, 3), (2,), (0,), 'C', False)

    def test_can_use_cub_nonsense_input3(self):
        if False:
            i = 10
            return i + 15
        a = cupy.random.random((3, 4, 5))
        a = a[:, 0:-1:2, 0:-1:3]
        assert not a.flags.forc
        b = cupy.empty((3,))
        assert self.can_use([a], [b], (1, 2), (0,)) is None

    def test_can_use_cub_zero_size_input(self):
        if False:
            while True:
                i = 10
        self._test_can_use((2, 0, 3), (), (0, 1, 2), (), 'C', False)

    def test_can_use_cub_oversize_input1(self):
        if False:
            print('Hello World!')
        mem = memory.alloc(100)
        a = cupy.ndarray((2 ** 6 * 1024 ** 3 + 1,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is None

    def test_can_use_cub_oversize_input2(self):
        if False:
            for i in range(10):
                print('nop')
        mem = memory.alloc(100)
        a = cupy.ndarray((2 ** 6 * 1024 ** 3,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is not None

    def test_can_use_cub_oversize_input3(self):
        if False:
            print('Hello World!')
        mem = memory.alloc(100)
        max_num = sys.maxsize
        a = cupy.ndarray((max_num,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is None

    def test_can_use_cub_oversize_input4(self):
        if False:
            print('Hello World!')
        mem = memory.alloc(100)
        a = cupy.ndarray((2 ** 31, 8), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (1,), (0,)) is None

    def test_can_use_accelerator_set_unset(self):
        if False:
            for i in range(10):
                print('nop')
        old_routine_accelerators = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators([])
        a = cupy.random.random((10, 10))
        func_name = ''.join(('cupy._core._cub_reduction.', '_SimpleCubReductionKernel_get_cached_function'))
        func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
        with testing.AssertFunctionIsCalled(func_name, wraps=func, times_called=2):
            a.sum()
        with testing.AssertFunctionIsCalled(func_name, wraps=func, times_called=1):
            a.sum(axis=1)
        with testing.AssertFunctionIsCalled(func_name, wraps=func, times_called=0):
            a.sum(axis=0)
        _accelerator.set_routine_accelerators(old_routine_accelerators)