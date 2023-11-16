import unittest
from unittest import mock
from cupy import cuda
from cupyx import profiler

@unittest.skipUnless(cuda.nvtx.available, 'nvtx is required for time_range')
class TestTimeRange(unittest.TestCase):

    def test_time_range(self):
        if False:
            i = 10
            return i + 15
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with profiler.time_range('test:time_range', color_id=-1):
                pass
            push.assert_called_once_with('test:time_range', -1)
            pop.assert_called_once_with()

    def test_time_range_with_ARGB(self):
        if False:
            print('Hello World!')
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with profiler.time_range('test:time_range_with_ARGB', argb_color=4278255360):
                pass
            push.assert_called_once_with('test:time_range_with_ARGB', 4278255360)
            pop.assert_called_once_with()

    def test_time_range_err(self):
        if False:
            i = 10
            return i + 15
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with profiler.time_range('test:time_range_error', -1):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:time_range_error', -1)
            pop.assert_called_once_with()

    def test_time_range_as_decorator(self):
        if False:
            while True:
                i = 10
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:

            @profiler.time_range()
            def f():
                if False:
                    i = 10
                    return i + 15
                pass
            f()
            push.assert_called_once_with('f', -1)
            pop.assert_called_once_with()

    def test_time_range_as_decorator_with_ARGB(self):
        if False:
            while True:
                i = 10
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:

            @profiler.time_range(argb_color=4294901760)
            def f():
                if False:
                    return 10
                pass
            f()
            push.assert_called_once_with('f', 4294901760)
            pop.assert_called_once_with()

    def test_time_range_as_decorator_err(self):
        if False:
            i = 10
            return i + 15
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:

            @profiler.time_range()
            def f():
                if False:
                    return 10
                raise Exception()
            try:
                f()
            except Exception:
                pass
            push.assert_called_once_with('f', -1)
            pop.assert_called_once_with()

class TestTimeRangeNVTXUnavailable(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.nvtx_available = cuda.nvtx.available
        cuda.nvtx.available = False

    def tearDown(self):
        if False:
            print('Hello World!')
        cuda.nvtx.available = self.nvtx_available

    def test_time_range(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(RuntimeError):
            with profiler.time_range(''):
                pass

    def test_time_range_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(RuntimeError):
            profiler.time_range()