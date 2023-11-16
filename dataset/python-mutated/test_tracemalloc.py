import contextlib
import os
import sys
import tracemalloc
import unittest
from unittest.mock import patch
from test.support.script_helper import assert_python_ok, assert_python_failure, interpreter_requires_environment
from test import support
from test.support import os_helper
try:
    import _testcapi
except ImportError:
    _testcapi = None
try:
    import cinderjit
except:
    cinderjit = None
EMPTY_STRING_SIZE = sys.getsizeof(b'')
INVALID_NFRAME = (-1, 2 ** 30)

def get_frames(nframe, lineno_delta):
    if False:
        i = 10
        return i + 15
    frames = []
    frame = sys._getframe(1)
    for index in range(nframe):
        code = frame.f_code
        lineno = frame.f_lineno + lineno_delta
        frames.append((code.co_filename, lineno))
        lineno_delta = 0
        frame = frame.f_back
        if frame is None:
            break
    return tuple(frames)

def allocate_bytes(size):
    if False:
        for i in range(10):
            print('nop')
    nframe = tracemalloc.get_traceback_limit()
    bytes_len = size - EMPTY_STRING_SIZE
    frames = get_frames(nframe, 1)
    data = b'x' * bytes_len
    return (data, tracemalloc.Traceback(frames, min(len(frames), nframe)))

def create_snapshots():
    if False:
        print('Hello World!')
    traceback_limit = 2
    raw_traces = [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (1, 2, (('a.py', 5), ('b.py', 4)), 3), (2, 66, (('b.py', 1),), 1), (3, 7, (('<unknown>', 0),), 1)]
    snapshot = tracemalloc.Snapshot(raw_traces, traceback_limit)
    raw_traces2 = [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (2, 2, (('a.py', 5), ('b.py', 4)), 3), (2, 5000, (('a.py', 5), ('b.py', 4)), 3), (4, 400, (('c.py', 578),), 1)]
    snapshot2 = tracemalloc.Snapshot(raw_traces2, traceback_limit)
    return (snapshot, snapshot2)

def frame(filename, lineno):
    if False:
        while True:
            i = 10
    return tracemalloc._Frame((filename, lineno))

def traceback(*frames):
    if False:
        while True:
            i = 10
    return tracemalloc.Traceback(frames)

def traceback_lineno(filename, lineno):
    if False:
        for i in range(10):
            print('nop')
    return traceback((filename, lineno))

def traceback_filename(filename):
    if False:
        while True:
            i = 10
    return traceback_lineno(filename, 0)

class TestTraceback(unittest.TestCase):

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')

        def get_repr(*args) -> str:
            if False:
                while True:
                    i = 10
            return repr(tracemalloc.Traceback(*args))
        self.assertEqual(get_repr(()), '<Traceback ()>')
        self.assertEqual(get_repr((), 0), '<Traceback () total_nframe=0>')
        frames = (('f1', 1), ('f2', 2))
        exp_repr_frames = "(<Frame filename='f2' lineno=2>, <Frame filename='f1' lineno=1>)"
        self.assertEqual(get_repr(frames), f'<Traceback {exp_repr_frames}>')
        self.assertEqual(get_repr(frames, 2), f'<Traceback {exp_repr_frames} total_nframe=2>')

class TestTracemallocEnabled(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if tracemalloc.is_tracing():
            self.skipTest('tracemalloc must be stopped before the test')
        tracemalloc.start(1)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        tracemalloc.stop()

    def test_get_tracemalloc_memory(self):
        if False:
            for i in range(10):
                print('nop')
        data = [allocate_bytes(123) for count in range(1000)]
        size = tracemalloc.get_tracemalloc_memory()
        self.assertGreaterEqual(size, 0)
        tracemalloc.clear_traces()
        size2 = tracemalloc.get_tracemalloc_memory()
        self.assertGreaterEqual(size2, 0)
        self.assertLessEqual(size2, size)

    def test_get_object_traceback(self):
        if False:
            i = 10
            return i + 15
        tracemalloc.clear_traces()
        obj_size = 12345
        (obj, obj_traceback) = allocate_bytes(obj_size)
        traceback = tracemalloc.get_object_traceback(obj)
        self.assertEqual(traceback, obj_traceback)

    def test_new_reference(self):
        if False:
            for i in range(10):
                print('nop')
        tracemalloc.clear_traces()
        support.gc_collect()
        obj = []
        obj = None
        obj = []
        nframe = tracemalloc.get_traceback_limit()
        frames = get_frames(nframe, -3)
        obj_traceback = tracemalloc.Traceback(frames, min(len(frames), nframe))
        traceback = tracemalloc.get_object_traceback(obj)
        self.assertIsNotNone(traceback)
        self.assertEqual(traceback, obj_traceback)

    def test_set_traceback_limit(self):
        if False:
            print('Hello World!')
        obj_size = 10
        tracemalloc.stop()
        self.assertRaises(ValueError, tracemalloc.start, -1)
        tracemalloc.stop()
        tracemalloc.start(10)
        (obj2, obj2_traceback) = allocate_bytes(obj_size)
        traceback = tracemalloc.get_object_traceback(obj2)
        self.assertEqual(len(traceback), 10)
        self.assertEqual(traceback, obj2_traceback)
        tracemalloc.stop()
        tracemalloc.start(1)
        (obj, obj_traceback) = allocate_bytes(obj_size)
        traceback = tracemalloc.get_object_traceback(obj)
        self.assertEqual(len(traceback), 1)
        self.assertEqual(traceback, obj_traceback)

    def find_trace(self, traces, traceback):
        if False:
            i = 10
            return i + 15
        for trace in traces:
            if trace[2] == traceback._frames:
                return trace
        self.fail('trace not found')

    def test_get_traces(self):
        if False:
            return 10
        tracemalloc.clear_traces()
        obj_size = 12345
        (obj, obj_traceback) = allocate_bytes(obj_size)
        traces = tracemalloc._get_traces()
        trace = self.find_trace(traces, obj_traceback)
        self.assertIsInstance(trace, tuple)
        (domain, size, traceback, length) = trace
        self.assertEqual(size, obj_size)
        self.assertEqual(traceback, obj_traceback._frames)
        tracemalloc.stop()
        self.assertEqual(tracemalloc._get_traces(), [])

    def test_get_traces_intern_traceback(self):
        if False:
            while True:
                i = 10

        def allocate_bytes2(size):
            if False:
                for i in range(10):
                    print('nop')
            return allocate_bytes(size)

        def allocate_bytes3(size):
            if False:
                i = 10
                return i + 15
            return allocate_bytes2(size)

        def allocate_bytes4(size):
            if False:
                while True:
                    i = 10
            return allocate_bytes3(size)
        tracemalloc.stop()
        tracemalloc.start(4)
        obj_size = 123
        (obj1, obj1_traceback) = allocate_bytes4(obj_size)
        (obj2, obj2_traceback) = allocate_bytes4(obj_size)
        traces = tracemalloc._get_traces()
        obj1_traceback._frames = tuple(reversed(obj1_traceback._frames))
        obj2_traceback._frames = tuple(reversed(obj2_traceback._frames))
        trace1 = self.find_trace(traces, obj1_traceback)
        trace2 = self.find_trace(traces, obj2_traceback)
        (domain1, size1, traceback1, length1) = trace1
        (domain2, size2, traceback2, length2) = trace2
        self.assertIs(traceback2, traceback1)

    @unittest.skipIf(cinderjit and cinderjit.is_inline_cache_stats_collection_enabled(), "#TODO(T150421262): Traced memory does not work well with JIT's inline cache stats collection.")
    def test_get_traced_memory(self):
        if False:
            for i in range(10):
                print('nop')
        max_error = 2048
        obj_size = 1024 * 1024
        tracemalloc.clear_traces()
        (obj, obj_traceback) = allocate_bytes(obj_size)
        (size, peak_size) = tracemalloc.get_traced_memory()
        self.assertGreaterEqual(size, obj_size)
        self.assertGreaterEqual(peak_size, size)
        self.assertLessEqual(size - obj_size, max_error)
        self.assertLessEqual(peak_size - size, max_error)
        obj = None
        (size2, peak_size2) = tracemalloc.get_traced_memory()
        self.assertLess(size2, size)
        self.assertGreaterEqual(size - size2, obj_size - max_error)
        self.assertGreaterEqual(peak_size2, peak_size)
        tracemalloc.clear_traces()
        self.assertEqual(tracemalloc.get_traced_memory(), (0, 0))
        (obj, obj_traceback) = allocate_bytes(obj_size)
        (size, peak_size) = tracemalloc.get_traced_memory()
        self.assertGreaterEqual(size, obj_size)
        tracemalloc.stop()
        self.assertEqual(tracemalloc.get_traced_memory(), (0, 0))

    def test_clear_traces(self):
        if False:
            return 10
        (obj, obj_traceback) = allocate_bytes(123)
        traceback = tracemalloc.get_object_traceback(obj)
        self.assertIsNotNone(traceback)
        tracemalloc.clear_traces()
        traceback2 = tracemalloc.get_object_traceback(obj)
        self.assertIsNone(traceback2)

    def test_reset_peak(self):
        if False:
            for i in range(10):
                print('nop')
        tracemalloc.clear_traces()
        large_sum = sum(list(range(100000)))
        (size1, peak1) = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        (size2, peak2) = tracemalloc.get_traced_memory()
        self.assertGreaterEqual(peak2, size2)
        self.assertLess(peak2, peak1)
        obj_size = 1024 * 1024
        (obj, obj_traceback) = allocate_bytes(obj_size)
        (size3, peak3) = tracemalloc.get_traced_memory()
        self.assertGreaterEqual(peak3, size3)
        self.assertGreater(peak3, peak2)
        self.assertGreaterEqual(peak3 - peak2, obj_size)

    def test_is_tracing(self):
        if False:
            for i in range(10):
                print('nop')
        tracemalloc.stop()
        self.assertFalse(tracemalloc.is_tracing())
        tracemalloc.start()
        self.assertTrue(tracemalloc.is_tracing())

    def test_snapshot(self):
        if False:
            print('Hello World!')
        (obj, source) = allocate_bytes(123)
        snapshot = tracemalloc.take_snapshot()
        self.assertGreater(snapshot.traces[1].traceback.total_nframe, 10)
        snapshot.dump(os_helper.TESTFN)
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        snapshot2 = tracemalloc.Snapshot.load(os_helper.TESTFN)
        self.assertEqual(snapshot2.traces, snapshot.traces)
        tracemalloc.stop()
        with self.assertRaises(RuntimeError) as cm:
            tracemalloc.take_snapshot()
        self.assertEqual(str(cm.exception), 'the tracemalloc module must be tracing memory allocations to take a snapshot')

    def test_snapshot_save_attr(self):
        if False:
            while True:
                i = 10
        snapshot = tracemalloc.take_snapshot()
        snapshot.test_attr = 'new'
        snapshot.dump(os_helper.TESTFN)
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        snapshot2 = tracemalloc.Snapshot.load(os_helper.TESTFN)
        self.assertEqual(snapshot2.test_attr, 'new')

    def fork_child(self):
        if False:
            print('Hello World!')
        if not tracemalloc.is_tracing():
            return 2
        obj_size = 12345
        (obj, obj_traceback) = allocate_bytes(obj_size)
        traceback = tracemalloc.get_object_traceback(obj)
        if traceback is None:
            return 3
        return 0

    @unittest.skipUnless(hasattr(os, 'fork'), 'need os.fork()')
    def test_fork(self):
        if False:
            for i in range(10):
                print('nop')
        pid = os.fork()
        if not pid:
            exitcode = 1
            try:
                exitcode = self.fork_child()
            finally:
                os._exit(exitcode)
        else:
            support.wait_process(pid, exitcode=0)

class TestSnapshot(unittest.TestCase):
    maxDiff = 4000

    def test_create_snapshot(self):
        if False:
            print('Hello World!')
        raw_traces = [(0, 5, (('a.py', 2),), 10)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(tracemalloc, 'is_tracing', return_value=True))
            stack.enter_context(patch.object(tracemalloc, 'get_traceback_limit', return_value=5))
            stack.enter_context(patch.object(tracemalloc, '_get_traces', return_value=raw_traces))
            snapshot = tracemalloc.take_snapshot()
            self.assertEqual(snapshot.traceback_limit, 5)
            self.assertEqual(len(snapshot.traces), 1)
            trace = snapshot.traces[0]
            self.assertEqual(trace.size, 5)
            self.assertEqual(trace.traceback.total_nframe, 10)
            self.assertEqual(len(trace.traceback), 1)
            self.assertEqual(trace.traceback[0].filename, 'a.py')
            self.assertEqual(trace.traceback[0].lineno, 2)

    def test_filter_traces(self):
        if False:
            i = 10
            return i + 15
        (snapshot, snapshot2) = create_snapshots()
        filter1 = tracemalloc.Filter(False, 'b.py')
        filter2 = tracemalloc.Filter(True, 'a.py', 2)
        filter3 = tracemalloc.Filter(True, 'a.py', 5)
        original_traces = list(snapshot.traces._traces)
        snapshot3 = snapshot.filter_traces((filter1,))
        self.assertEqual(snapshot3.traces._traces, [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (1, 2, (('a.py', 5), ('b.py', 4)), 3), (3, 7, (('<unknown>', 0),), 1)])
        self.assertEqual(snapshot.traces._traces, original_traces)
        snapshot4 = snapshot3.filter_traces((filter2, filter3))
        self.assertEqual(snapshot4.traces._traces, [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (1, 2, (('a.py', 5), ('b.py', 4)), 3)])
        snapshot5 = snapshot.filter_traces(())
        self.assertIsNot(snapshot5, snapshot)
        self.assertIsNot(snapshot5.traces, snapshot.traces)
        self.assertEqual(snapshot5.traces, snapshot.traces)
        self.assertRaises(TypeError, snapshot.filter_traces, filter1)

    def test_filter_traces_domain(self):
        if False:
            print('Hello World!')
        (snapshot, snapshot2) = create_snapshots()
        filter1 = tracemalloc.Filter(False, 'a.py', domain=1)
        filter2 = tracemalloc.Filter(True, 'a.py', domain=1)
        original_traces = list(snapshot.traces._traces)
        snapshot3 = snapshot.filter_traces((filter1,))
        self.assertEqual(snapshot3.traces._traces, [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (2, 66, (('b.py', 1),), 1), (3, 7, (('<unknown>', 0),), 1)])
        snapshot3 = snapshot.filter_traces((filter1,))
        self.assertEqual(snapshot3.traces._traces, [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (2, 66, (('b.py', 1),), 1), (3, 7, (('<unknown>', 0),), 1)])

    def test_filter_traces_domain_filter(self):
        if False:
            for i in range(10):
                print('nop')
        (snapshot, snapshot2) = create_snapshots()
        filter1 = tracemalloc.DomainFilter(False, domain=3)
        filter2 = tracemalloc.DomainFilter(True, domain=3)
        snapshot3 = snapshot.filter_traces((filter1,))
        self.assertEqual(snapshot3.traces._traces, [(0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (0, 10, (('a.py', 2), ('b.py', 4)), 3), (1, 2, (('a.py', 5), ('b.py', 4)), 3), (2, 66, (('b.py', 1),), 1)])
        snapshot3 = snapshot.filter_traces((filter2,))
        self.assertEqual(snapshot3.traces._traces, [(3, 7, (('<unknown>', 0),), 1)])

    def test_snapshot_group_by_line(self):
        if False:
            i = 10
            return i + 15
        (snapshot, snapshot2) = create_snapshots()
        tb_0 = traceback_lineno('<unknown>', 0)
        tb_a_2 = traceback_lineno('a.py', 2)
        tb_a_5 = traceback_lineno('a.py', 5)
        tb_b_1 = traceback_lineno('b.py', 1)
        tb_c_578 = traceback_lineno('c.py', 578)
        stats1 = snapshot.statistics('lineno')
        self.assertEqual(stats1, [tracemalloc.Statistic(tb_b_1, 66, 1), tracemalloc.Statistic(tb_a_2, 30, 3), tracemalloc.Statistic(tb_0, 7, 1), tracemalloc.Statistic(tb_a_5, 2, 1)])
        stats2 = snapshot2.statistics('lineno')
        self.assertEqual(stats2, [tracemalloc.Statistic(tb_a_5, 5002, 2), tracemalloc.Statistic(tb_c_578, 400, 1), tracemalloc.Statistic(tb_a_2, 30, 3)])
        statistics = snapshot2.compare_to(snapshot, 'lineno')
        self.assertEqual(statistics, [tracemalloc.StatisticDiff(tb_a_5, 5002, 5000, 2, 1), tracemalloc.StatisticDiff(tb_c_578, 400, 400, 1, 1), tracemalloc.StatisticDiff(tb_b_1, 0, -66, 0, -1), tracemalloc.StatisticDiff(tb_0, 0, -7, 0, -1), tracemalloc.StatisticDiff(tb_a_2, 30, 0, 3, 0)])

    def test_snapshot_group_by_file(self):
        if False:
            while True:
                i = 10
        (snapshot, snapshot2) = create_snapshots()
        tb_0 = traceback_filename('<unknown>')
        tb_a = traceback_filename('a.py')
        tb_b = traceback_filename('b.py')
        tb_c = traceback_filename('c.py')
        stats1 = snapshot.statistics('filename')
        self.assertEqual(stats1, [tracemalloc.Statistic(tb_b, 66, 1), tracemalloc.Statistic(tb_a, 32, 4), tracemalloc.Statistic(tb_0, 7, 1)])
        stats2 = snapshot2.statistics('filename')
        self.assertEqual(stats2, [tracemalloc.Statistic(tb_a, 5032, 5), tracemalloc.Statistic(tb_c, 400, 1)])
        diff = snapshot2.compare_to(snapshot, 'filename')
        self.assertEqual(diff, [tracemalloc.StatisticDiff(tb_a, 5032, 5000, 5, 1), tracemalloc.StatisticDiff(tb_c, 400, 400, 1, 1), tracemalloc.StatisticDiff(tb_b, 0, -66, 0, -1), tracemalloc.StatisticDiff(tb_0, 0, -7, 0, -1)])

    def test_snapshot_group_by_traceback(self):
        if False:
            return 10
        (snapshot, snapshot2) = create_snapshots()
        tb1 = traceback(('a.py', 2), ('b.py', 4))
        tb2 = traceback(('a.py', 5), ('b.py', 4))
        tb3 = traceback(('b.py', 1))
        tb4 = traceback(('<unknown>', 0))
        stats1 = snapshot.statistics('traceback')
        self.assertEqual(stats1, [tracemalloc.Statistic(tb3, 66, 1), tracemalloc.Statistic(tb1, 30, 3), tracemalloc.Statistic(tb4, 7, 1), tracemalloc.Statistic(tb2, 2, 1)])
        tb5 = traceback(('c.py', 578))
        stats2 = snapshot2.statistics('traceback')
        self.assertEqual(stats2, [tracemalloc.Statistic(tb2, 5002, 2), tracemalloc.Statistic(tb5, 400, 1), tracemalloc.Statistic(tb1, 30, 3)])
        diff = snapshot2.compare_to(snapshot, 'traceback')
        self.assertEqual(diff, [tracemalloc.StatisticDiff(tb2, 5002, 5000, 2, 1), tracemalloc.StatisticDiff(tb5, 400, 400, 1, 1), tracemalloc.StatisticDiff(tb3, 0, -66, 0, -1), tracemalloc.StatisticDiff(tb4, 0, -7, 0, -1), tracemalloc.StatisticDiff(tb1, 30, 0, 3, 0)])
        self.assertRaises(ValueError, snapshot.statistics, 'traceback', cumulative=True)

    def test_snapshot_group_by_cumulative(self):
        if False:
            i = 10
            return i + 15
        (snapshot, snapshot2) = create_snapshots()
        tb_0 = traceback_filename('<unknown>')
        tb_a = traceback_filename('a.py')
        tb_b = traceback_filename('b.py')
        tb_a_2 = traceback_lineno('a.py', 2)
        tb_a_5 = traceback_lineno('a.py', 5)
        tb_b_1 = traceback_lineno('b.py', 1)
        tb_b_4 = traceback_lineno('b.py', 4)
        stats = snapshot.statistics('filename', True)
        self.assertEqual(stats, [tracemalloc.Statistic(tb_b, 98, 5), tracemalloc.Statistic(tb_a, 32, 4), tracemalloc.Statistic(tb_0, 7, 1)])
        stats = snapshot.statistics('lineno', True)
        self.assertEqual(stats, [tracemalloc.Statistic(tb_b_1, 66, 1), tracemalloc.Statistic(tb_b_4, 32, 4), tracemalloc.Statistic(tb_a_2, 30, 3), tracemalloc.Statistic(tb_0, 7, 1), tracemalloc.Statistic(tb_a_5, 2, 1)])

    def test_trace_format(self):
        if False:
            i = 10
            return i + 15
        (snapshot, snapshot2) = create_snapshots()
        trace = snapshot.traces[0]
        self.assertEqual(str(trace), 'b.py:4: 10 B')
        traceback = trace.traceback
        self.assertEqual(str(traceback), 'b.py:4')
        frame = traceback[0]
        self.assertEqual(str(frame), 'b.py:4')

    def test_statistic_format(self):
        if False:
            print('Hello World!')
        (snapshot, snapshot2) = create_snapshots()
        stats = snapshot.statistics('lineno')
        stat = stats[0]
        self.assertEqual(str(stat), 'b.py:1: size=66 B, count=1, average=66 B')

    def test_statistic_diff_format(self):
        if False:
            i = 10
            return i + 15
        (snapshot, snapshot2) = create_snapshots()
        stats = snapshot2.compare_to(snapshot, 'lineno')
        stat = stats[0]
        self.assertEqual(str(stat), 'a.py:5: size=5002 B (+5000 B), count=2 (+1), average=2501 B')

    def test_slices(self):
        if False:
            print('Hello World!')
        (snapshot, snapshot2) = create_snapshots()
        self.assertEqual(snapshot.traces[:2], (snapshot.traces[0], snapshot.traces[1]))
        traceback = snapshot.traces[0].traceback
        self.assertEqual(traceback[:2], (traceback[0], traceback[1]))

    def test_format_traceback(self):
        if False:
            return 10
        (snapshot, snapshot2) = create_snapshots()

        def getline(filename, lineno):
            if False:
                print('Hello World!')
            return '  <%s, %s>' % (filename, lineno)
        with unittest.mock.patch('tracemalloc.linecache.getline', side_effect=getline):
            tb = snapshot.traces[0].traceback
            self.assertEqual(tb.format(), ['  File "b.py", line 4', '    <b.py, 4>', '  File "a.py", line 2', '    <a.py, 2>'])
            self.assertEqual(tb.format(limit=1), ['  File "a.py", line 2', '    <a.py, 2>'])
            self.assertEqual(tb.format(limit=-1), ['  File "b.py", line 4', '    <b.py, 4>'])
            self.assertEqual(tb.format(most_recent_first=True), ['  File "a.py", line 2', '    <a.py, 2>', '  File "b.py", line 4', '    <b.py, 4>'])
            self.assertEqual(tb.format(limit=1, most_recent_first=True), ['  File "a.py", line 2', '    <a.py, 2>'])
            self.assertEqual(tb.format(limit=-1, most_recent_first=True), ['  File "b.py", line 4', '    <b.py, 4>'])

class TestFilters(unittest.TestCase):
    maxDiff = 2048

    def test_filter_attributes(self):
        if False:
            return 10
        f = tracemalloc.Filter(True, 'abc')
        self.assertEqual(f.inclusive, True)
        self.assertEqual(f.filename_pattern, 'abc')
        self.assertIsNone(f.lineno)
        self.assertEqual(f.all_frames, False)
        f = tracemalloc.Filter(False, 'test.py', 123, True)
        self.assertEqual(f.inclusive, False)
        self.assertEqual(f.filename_pattern, 'test.py')
        self.assertEqual(f.lineno, 123)
        self.assertEqual(f.all_frames, True)
        f = tracemalloc.Filter(inclusive=False, filename_pattern='test.py', lineno=123, all_frames=True)
        self.assertEqual(f.inclusive, False)
        self.assertEqual(f.filename_pattern, 'test.py')
        self.assertEqual(f.lineno, 123)
        self.assertEqual(f.all_frames, True)
        self.assertRaises(AttributeError, setattr, f, 'filename_pattern', 'abc')

    def test_filter_match(self):
        if False:
            i = 10
            return i + 15
        f = tracemalloc.Filter(True, 'abc')
        self.assertTrue(f._match_frame('abc', 0))
        self.assertTrue(f._match_frame('abc', 5))
        self.assertTrue(f._match_frame('abc', 10))
        self.assertFalse(f._match_frame('12356', 0))
        self.assertFalse(f._match_frame('12356', 5))
        self.assertFalse(f._match_frame('12356', 10))
        f = tracemalloc.Filter(False, 'abc')
        self.assertFalse(f._match_frame('abc', 0))
        self.assertFalse(f._match_frame('abc', 5))
        self.assertFalse(f._match_frame('abc', 10))
        self.assertTrue(f._match_frame('12356', 0))
        self.assertTrue(f._match_frame('12356', 5))
        self.assertTrue(f._match_frame('12356', 10))
        f = tracemalloc.Filter(True, 'abc', 5)
        self.assertFalse(f._match_frame('abc', 0))
        self.assertTrue(f._match_frame('abc', 5))
        self.assertFalse(f._match_frame('abc', 10))
        self.assertFalse(f._match_frame('12356', 0))
        self.assertFalse(f._match_frame('12356', 5))
        self.assertFalse(f._match_frame('12356', 10))
        f = tracemalloc.Filter(False, 'abc', 5)
        self.assertTrue(f._match_frame('abc', 0))
        self.assertFalse(f._match_frame('abc', 5))
        self.assertTrue(f._match_frame('abc', 10))
        self.assertTrue(f._match_frame('12356', 0))
        self.assertTrue(f._match_frame('12356', 5))
        self.assertTrue(f._match_frame('12356', 10))
        f = tracemalloc.Filter(True, 'abc', 0)
        self.assertTrue(f._match_frame('abc', 0))
        self.assertFalse(f._match_frame('abc', 5))
        self.assertFalse(f._match_frame('abc', 10))
        self.assertFalse(f._match_frame('12356', 0))
        self.assertFalse(f._match_frame('12356', 5))
        self.assertFalse(f._match_frame('12356', 10))
        f = tracemalloc.Filter(False, 'abc', 0)
        self.assertFalse(f._match_frame('abc', 0))
        self.assertTrue(f._match_frame('abc', 5))
        self.assertTrue(f._match_frame('abc', 10))
        self.assertTrue(f._match_frame('12356', 0))
        self.assertTrue(f._match_frame('12356', 5))
        self.assertTrue(f._match_frame('12356', 10))

    def test_filter_match_filename(self):
        if False:
            i = 10
            return i + 15

        def fnmatch(inclusive, filename, pattern):
            if False:
                return 10
            f = tracemalloc.Filter(inclusive, pattern)
            return f._match_frame(filename, 0)
        self.assertTrue(fnmatch(True, 'abc', 'abc'))
        self.assertFalse(fnmatch(True, '12356', 'abc'))
        self.assertFalse(fnmatch(True, '<unknown>', 'abc'))
        self.assertFalse(fnmatch(False, 'abc', 'abc'))
        self.assertTrue(fnmatch(False, '12356', 'abc'))
        self.assertTrue(fnmatch(False, '<unknown>', 'abc'))

    def test_filter_match_filename_joker(self):
        if False:
            print('Hello World!')

        def fnmatch(filename, pattern):
            if False:
                return 10
            filter = tracemalloc.Filter(True, pattern)
            return filter._match_frame(filename, 0)
        self.assertFalse(fnmatch('abc', ''))
        self.assertFalse(fnmatch('', 'abc'))
        self.assertTrue(fnmatch('', ''))
        self.assertTrue(fnmatch('', '*'))
        self.assertTrue(fnmatch('abc', 'abc'))
        self.assertFalse(fnmatch('abc', 'abcd'))
        self.assertFalse(fnmatch('abc', 'def'))
        self.assertTrue(fnmatch('abc', 'a*'))
        self.assertTrue(fnmatch('abc', 'abc*'))
        self.assertFalse(fnmatch('abc', 'b*'))
        self.assertFalse(fnmatch('abc', 'abcd*'))
        self.assertTrue(fnmatch('abc', 'a*c'))
        self.assertTrue(fnmatch('abcdcx', 'a*cx'))
        self.assertFalse(fnmatch('abb', 'a*c'))
        self.assertFalse(fnmatch('abcdce', 'a*cx'))
        self.assertTrue(fnmatch('abcde', 'a*c*e'))
        self.assertTrue(fnmatch('abcbdefeg', 'a*bd*eg'))
        self.assertFalse(fnmatch('abcdd', 'a*c*e'))
        self.assertFalse(fnmatch('abcbdefef', 'a*bd*eg'))
        self.assertTrue(fnmatch('a.pyc', 'a.py'))
        self.assertTrue(fnmatch('a.py', 'a.pyc'))
        if os.name == 'nt':
            self.assertTrue(fnmatch('aBC', 'ABc'))
            self.assertTrue(fnmatch('aBcDe', 'Ab*dE'))
            self.assertTrue(fnmatch('a.pyc', 'a.PY'))
            self.assertTrue(fnmatch('a.py', 'a.PYC'))
        else:
            self.assertFalse(fnmatch('aBC', 'ABc'))
            self.assertFalse(fnmatch('aBcDe', 'Ab*dE'))
            self.assertFalse(fnmatch('a.pyc', 'a.PY'))
            self.assertFalse(fnmatch('a.py', 'a.PYC'))
        if os.name == 'nt':
            self.assertTrue(fnmatch('a/b', 'a\\b'))
            self.assertTrue(fnmatch('a\\b', 'a/b'))
            self.assertTrue(fnmatch('a/b\\c', 'a\\b/c'))
            self.assertTrue(fnmatch('a/b/c', 'a\\b\\c'))
        else:
            self.assertFalse(fnmatch('a/b', 'a\\b'))
            self.assertFalse(fnmatch('a\\b', 'a/b'))
            self.assertFalse(fnmatch('a/b\\c', 'a\\b/c'))
            self.assertFalse(fnmatch('a/b/c', 'a\\b\\c'))
        self.assertFalse(fnmatch('a.pyo', 'a.py'))

    def test_filter_match_trace(self):
        if False:
            while True:
                i = 10
        t1 = (('a.py', 2), ('b.py', 3))
        t2 = (('b.py', 4), ('b.py', 5))
        t3 = (('c.py', 5), ('<unknown>', 0))
        unknown = (('<unknown>', 0),)
        f = tracemalloc.Filter(True, 'b.py', all_frames=True)
        self.assertTrue(f._match_traceback(t1))
        self.assertTrue(f._match_traceback(t2))
        self.assertFalse(f._match_traceback(t3))
        self.assertFalse(f._match_traceback(unknown))
        f = tracemalloc.Filter(True, 'b.py', all_frames=False)
        self.assertFalse(f._match_traceback(t1))
        self.assertTrue(f._match_traceback(t2))
        self.assertFalse(f._match_traceback(t3))
        self.assertFalse(f._match_traceback(unknown))
        f = tracemalloc.Filter(False, 'b.py', all_frames=True)
        self.assertFalse(f._match_traceback(t1))
        self.assertFalse(f._match_traceback(t2))
        self.assertTrue(f._match_traceback(t3))
        self.assertTrue(f._match_traceback(unknown))
        f = tracemalloc.Filter(False, 'b.py', all_frames=False)
        self.assertTrue(f._match_traceback(t1))
        self.assertFalse(f._match_traceback(t2))
        self.assertTrue(f._match_traceback(t3))
        self.assertTrue(f._match_traceback(unknown))
        f = tracemalloc.Filter(False, '<unknown>', all_frames=False)
        self.assertTrue(f._match_traceback(t1))
        self.assertTrue(f._match_traceback(t2))
        self.assertTrue(f._match_traceback(t3))
        self.assertFalse(f._match_traceback(unknown))
        f = tracemalloc.Filter(True, '<unknown>', all_frames=True)
        self.assertFalse(f._match_traceback(t1))
        self.assertFalse(f._match_traceback(t2))
        self.assertTrue(f._match_traceback(t3))
        self.assertTrue(f._match_traceback(unknown))
        f = tracemalloc.Filter(False, '<unknown>', all_frames=True)
        self.assertTrue(f._match_traceback(t1))
        self.assertTrue(f._match_traceback(t2))
        self.assertFalse(f._match_traceback(t3))
        self.assertFalse(f._match_traceback(unknown))

class TestCommandLine(unittest.TestCase):

    def test_env_var_disabled_by_default(self):
        if False:
            print('Hello World!')
        code = 'import tracemalloc; print(tracemalloc.is_tracing())'
        (ok, stdout, stderr) = assert_python_ok('-c', code)
        stdout = stdout.rstrip()
        self.assertEqual(stdout, b'False')

    @unittest.skipIf(interpreter_requires_environment(), 'Cannot run -E tests when PYTHON env vars are required.')
    def test_env_var_ignored_with_E(self):
        if False:
            i = 10
            return i + 15
        'PYTHON* environment variables must be ignored when -E is present.'
        code = 'import tracemalloc; print(tracemalloc.is_tracing())'
        (ok, stdout, stderr) = assert_python_ok('-E', '-c', code, PYTHONTRACEMALLOC='1')
        stdout = stdout.rstrip()
        self.assertEqual(stdout, b'False')

    def test_env_var_disabled(self):
        if False:
            i = 10
            return i + 15
        code = 'import tracemalloc; print(tracemalloc.is_tracing())'
        (ok, stdout, stderr) = assert_python_ok('-c', code, PYTHONTRACEMALLOC='0')
        stdout = stdout.rstrip()
        self.assertEqual(stdout, b'False')

    def test_env_var_enabled_at_startup(self):
        if False:
            while True:
                i = 10
        code = 'import tracemalloc; print(tracemalloc.is_tracing())'
        (ok, stdout, stderr) = assert_python_ok('-c', code, PYTHONTRACEMALLOC='1')
        stdout = stdout.rstrip()
        self.assertEqual(stdout, b'True')

    def test_env_limit(self):
        if False:
            print('Hello World!')
        code = 'import tracemalloc; print(tracemalloc.get_traceback_limit())'
        (ok, stdout, stderr) = assert_python_ok('-c', code, PYTHONTRACEMALLOC='10')
        stdout = stdout.rstrip()
        self.assertEqual(stdout, b'10')

    def check_env_var_invalid(self, nframe):
        if False:
            return 10
        with support.SuppressCrashReport():
            (ok, stdout, stderr) = assert_python_failure('-c', 'pass', PYTHONTRACEMALLOC=str(nframe))
        if b'ValueError: the number of frames must be in range' in stderr:
            return
        if b'PYTHONTRACEMALLOC: invalid number of frames' in stderr:
            return
        self.fail(f'unexpected output: {stderr!a}')

    def test_env_var_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        for nframe in INVALID_NFRAME:
            with self.subTest(nframe=nframe):
                self.check_env_var_invalid(nframe)

    def test_sys_xoptions(self):
        if False:
            i = 10
            return i + 15
        for (xoptions, nframe) in (('tracemalloc', 1), ('tracemalloc=1', 1), ('tracemalloc=15', 15)):
            with self.subTest(xoptions=xoptions, nframe=nframe):
                code = 'import tracemalloc; print(tracemalloc.get_traceback_limit())'
                (ok, stdout, stderr) = assert_python_ok('-X', xoptions, '-c', code)
                stdout = stdout.rstrip()
                self.assertEqual(stdout, str(nframe).encode('ascii'))

    def check_sys_xoptions_invalid(self, nframe):
        if False:
            print('Hello World!')
        args = ('-X', 'tracemalloc=%s' % nframe, '-c', 'pass')
        with support.SuppressCrashReport():
            (ok, stdout, stderr) = assert_python_failure(*args)
        if b'ValueError: the number of frames must be in range' in stderr:
            return
        if b'-X tracemalloc=NFRAME: invalid number of frames' in stderr:
            return
        self.fail(f'unexpected output: {stderr!a}')

    def test_sys_xoptions_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        for nframe in INVALID_NFRAME:
            with self.subTest(nframe=nframe):
                self.check_sys_xoptions_invalid(nframe)

    @unittest.skipIf(_testcapi is None, 'need _testcapi')
    def test_pymem_alloc0(self):
        if False:
            print('Hello World!')
        code = 'import _testcapi; _testcapi.test_pymem_alloc0(); 1'
        assert_python_ok('-X', 'tracemalloc', '-c', code)

@unittest.skipIf(_testcapi is None, 'need _testcapi')
class TestCAPI(unittest.TestCase):
    maxDiff = 80 * 20

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if tracemalloc.is_tracing():
            self.skipTest('tracemalloc must be stopped before the test')
        self.domain = 5
        self.size = 123
        self.obj = allocate_bytes(self.size)[0]
        self.ptr = id(self.obj)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        tracemalloc.stop()

    def get_traceback(self):
        if False:
            for i in range(10):
                print('nop')
        frames = _testcapi.tracemalloc_get_traceback(self.domain, self.ptr)
        if frames is not None:
            return tracemalloc.Traceback(frames)
        else:
            return None

    def track(self, release_gil=False, nframe=1):
        if False:
            i = 10
            return i + 15
        frames = get_frames(nframe, 1)
        _testcapi.tracemalloc_track(self.domain, self.ptr, self.size, release_gil)
        return frames

    def untrack(self):
        if False:
            while True:
                i = 10
        _testcapi.tracemalloc_untrack(self.domain, self.ptr)

    def get_traced_memory(self):
        if False:
            i = 10
            return i + 15
        snapshot = tracemalloc.take_snapshot()
        domain_filter = tracemalloc.DomainFilter(True, self.domain)
        snapshot = snapshot.filter_traces([domain_filter])
        return sum((trace.size for trace in snapshot.traces))

    def check_track(self, release_gil):
        if False:
            while True:
                i = 10
        nframe = 5
        tracemalloc.start(nframe)
        size = tracemalloc.get_traced_memory()[0]
        frames = self.track(release_gil, nframe)
        self.assertEqual(self.get_traceback(), tracemalloc.Traceback(frames))
        self.assertEqual(self.get_traced_memory(), self.size)

    def test_track(self):
        if False:
            while True:
                i = 10
        self.check_track(False)

    def test_track_without_gil(self):
        if False:
            while True:
                i = 10
        self.check_track(True)

    def test_track_already_tracked(self):
        if False:
            while True:
                i = 10
        nframe = 5
        tracemalloc.start(nframe)
        self.track()
        frames = self.track(nframe=nframe)
        self.assertEqual(self.get_traceback(), tracemalloc.Traceback(frames))

    def test_untrack(self):
        if False:
            while True:
                i = 10
        tracemalloc.start()
        self.track()
        self.assertIsNotNone(self.get_traceback())
        self.assertEqual(self.get_traced_memory(), self.size)
        self.untrack()
        self.assertIsNone(self.get_traceback())
        self.assertEqual(self.get_traced_memory(), 0)
        self.untrack()
        self.untrack()

    def test_stop_track(self):
        if False:
            i = 10
            return i + 15
        tracemalloc.start()
        tracemalloc.stop()
        with self.assertRaises(RuntimeError):
            self.track()
        self.assertIsNone(self.get_traceback())

    def test_stop_untrack(self):
        if False:
            while True:
                i = 10
        tracemalloc.start()
        self.track()
        tracemalloc.stop()
        with self.assertRaises(RuntimeError):
            self.untrack()
if __name__ == '__main__':
    unittest.main()