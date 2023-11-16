from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, skip_if_external_memmgr, CUDATestCase
from numba.tests.support import captured_stderr
from numba.core import config

@skip_on_cudasim('not supported on CUDASIM')
@skip_if_external_memmgr('Deallocation specific to Numba memory management')
class TestDeallocation(CUDATestCase):

    def test_max_pending_count(self):
        if False:
            for i in range(10):
                print('nop')
        deallocs = cuda.current_context().memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        for i in range(config.CUDA_DEALLOCS_COUNT):
            cuda.to_device(np.arange(1))
            self.assertEqual(len(deallocs), i + 1)
        cuda.to_device(np.arange(1))
        self.assertEqual(len(deallocs), 0)

    def test_max_pending_bytes(self):
        if False:
            i = 10
            return i + 15
        ctx = cuda.current_context()
        deallocs = ctx.memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        mi = ctx.get_memory_info()
        max_pending = 10 ** 6
        old_ratio = config.CUDA_DEALLOCS_RATIO
        try:
            config.CUDA_DEALLOCS_RATIO = max_pending / mi.total
            self.assertAlmostEqual(deallocs._max_pending_bytes, max_pending, delta=1)
            cuda.to_device(np.ones(max_pending // 2, dtype=np.int8))
            self.assertEqual(len(deallocs), 1)
            cuda.to_device(np.ones(deallocs._max_pending_bytes - deallocs._size, dtype=np.int8))
            self.assertEqual(len(deallocs), 2)
            cuda.to_device(np.ones(1, dtype=np.int8))
            self.assertEqual(len(deallocs), 0)
        finally:
            config.CUDA_DEALLOCS_RATIO = old_ratio

@skip_on_cudasim('defer_cleanup has no effect in CUDASIM')
@skip_if_external_memmgr('Deallocation specific to Numba memory management')
class TestDeferCleanup(CUDATestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        with cuda.defer_cleanup():
            darr2 = cuda.to_device(harr)
            del darr1
            self.assertEqual(len(deallocs), 1)
            del darr2
            self.assertEqual(len(deallocs), 2)
            deallocs.clear()
            self.assertEqual(len(deallocs), 2)
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

    def test_nested(self):
        if False:
            return 10
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        with cuda.defer_cleanup():
            with cuda.defer_cleanup():
                darr2 = cuda.to_device(harr)
                del darr1
                self.assertEqual(len(deallocs), 1)
                del darr2
                self.assertEqual(len(deallocs), 2)
                deallocs.clear()
                self.assertEqual(len(deallocs), 2)
            deallocs.clear()
            self.assertEqual(len(deallocs), 2)
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

        class CustomError(Exception):
            pass
        with self.assertRaises(CustomError):
            with cuda.defer_cleanup():
                darr2 = cuda.to_device(harr)
                del darr2
                self.assertEqual(len(deallocs), 1)
                deallocs.clear()
                self.assertEqual(len(deallocs), 1)
                raise CustomError
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        del darr1
        self.assertEqual(len(deallocs), 1)
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

class TestDeferCleanupAvail(CUDATestCase):

    def test_context_manager(self):
        if False:
            print('Hello World!')
        with cuda.defer_cleanup():
            pass

@skip_on_cudasim('not supported on CUDASIM')
class TestDel(CUDATestCase):
    """
    Ensure resources are deleted properly without ignored exception.
    """

    @contextmanager
    def check_ignored_exception(self, ctx):
        if False:
            print('Hello World!')
        with captured_stderr() as cap:
            yield
            ctx.deallocations.clear()
        self.assertFalse(cap.getvalue())

    def test_stream(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = cuda.current_context()
        stream = ctx.create_stream()
        with self.check_ignored_exception(ctx):
            del stream

    def test_event(self):
        if False:
            while True:
                i = 10
        ctx = cuda.current_context()
        event = ctx.create_event()
        with self.check_ignored_exception(ctx):
            del event

    def test_pinned_memory(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_mapped_memory(self):
        if False:
            print('Hello World!')
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32, mapped=True)
        with self.check_ignored_exception(ctx):
            del mem

    def test_device_memory(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = cuda.current_context()
        mem = ctx.memalloc(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_managed_memory(self):
        if False:
            print('Hello World!')
        ctx = cuda.current_context()
        mem = ctx.memallocmanaged(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_pinned_contextmanager(self):
        if False:
            print('Hello World!')

        class PinnedException(Exception):
            pass
        arr = np.zeros(1)
        ctx = cuda.current_context()
        ctx.deallocations.clear()
        with self.check_ignored_exception(ctx):
            with cuda.pinned(arr):
                pass
            with cuda.pinned(arr):
                pass
            with cuda.defer_cleanup():
                with cuda.pinned(arr):
                    pass
                with cuda.pinned(arr):
                    pass
            try:
                with cuda.pinned(arr):
                    raise PinnedException
            except PinnedException:
                with cuda.pinned(arr):
                    pass

    def test_mapped_contextmanager(self):
        if False:
            for i in range(10):
                print('nop')

        class MappedException(Exception):
            pass
        arr = np.zeros(1)
        ctx = cuda.current_context()
        ctx.deallocations.clear()
        with self.check_ignored_exception(ctx):
            with cuda.mapped(arr):
                pass
            with cuda.mapped(arr):
                pass
            with cuda.defer_cleanup():
                with cuda.mapped(arr):
                    pass
                with cuda.mapped(arr):
                    pass
            try:
                with cuda.mapped(arr):
                    raise MappedException
            except MappedException:
                with cuda.mapped(arr):
                    pass
if __name__ == '__main__':
    unittest.main()