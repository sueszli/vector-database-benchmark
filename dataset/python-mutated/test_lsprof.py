"""Tests for profiling data collection."""
import cPickle
import threading
import bzrlib
from bzrlib import errors, tests
from bzrlib.tests import features
_TXT_HEADER = '   CallCount    Recursive    Total(ms)   ' + 'Inline(ms) module:lineno(function)\n'

def _junk_callable():
    if False:
        print('Hello World!')
    'A simple routine to profile.'
    result = sorted(['abc', 'def', 'ghi'])

def _collect_stats():
    if False:
        for i in range(10):
            print('nop')
    'Collect and return some dummy profile data.'
    from bzrlib.lsprof import profile
    (ret, stats) = profile(_junk_callable)
    return stats

class TestStatsSave(tests.TestCaseInTempDir):
    _test_needs_features = [features.lsprof_feature]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(tests.TestCaseInTempDir, self).setUp()
        self.stats = _collect_stats()

    def _tempfile(self, ext):
        if False:
            i = 10
            return i + 15
        dir = self.test_dir
        return bzrlib.osutils.pathjoin(dir, 'tmp_profile_data.' + ext)

    def test_stats_save_to_txt(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._tempfile('txt')
        self.stats.save(f)
        lines = open(f).readlines()
        self.assertEqual(lines[0], _TXT_HEADER)

    def test_stats_save_to_callgrind(self):
        if False:
            return 10
        f = self._tempfile('callgrind')
        self.stats.save(f)
        lines = open(f).readlines()
        self.assertEqual(lines[0], 'events: Ticks\n')
        f = bzrlib.osutils.pathjoin(self.test_dir, 'callgrind.out.foo')
        self.stats.save(f)
        lines = open(f).readlines()
        self.assertEqual(lines[0], 'events: Ticks\n')
        f2 = self._tempfile('txt')
        self.stats.save(f2, format='callgrind')
        lines2 = open(f2).readlines()
        self.assertEqual(lines2[0], 'events: Ticks\n')

    def test_stats_save_to_pickle(self):
        if False:
            print('Hello World!')
        f = self._tempfile('pkl')
        self.stats.save(f)
        data1 = cPickle.load(open(f))
        self.assertEqual(type(data1), bzrlib.lsprof.Stats)

class TestBzrProfiler(tests.TestCase):
    _test_needs_features = [features.lsprof_feature]

    def test_start_call_stuff_stop(self):
        if False:
            for i in range(10):
                print('nop')
        profiler = bzrlib.lsprof.BzrProfiler()
        profiler.start()
        try:

            def a_function():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            a_function()
        finally:
            stats = profiler.stop()
        stats.freeze()
        lines = [str(data) for data in stats.data]
        lines = [line for line in lines if 'a_function' in line]
        self.assertLength(1, lines)

    def test_block_0(self):
        if False:
            for i in range(10):
                print('nop')
        self.overrideAttr(bzrlib.lsprof.BzrProfiler, 'profiler_block', 0)
        inner_calls = []

        def inner():
            if False:
                while True:
                    i = 10
            profiler = bzrlib.lsprof.BzrProfiler()
            self.assertRaises(errors.BzrError, profiler.start)
            inner_calls.append(True)
        bzrlib.lsprof.profile(inner)
        self.assertLength(1, inner_calls)

    def test_block_1(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def profiled():
            if False:
                for i in range(10):
                    print('nop')
            calls.append('profiled')

        def do_profile():
            if False:
                i = 10
                return i + 15
            bzrlib.lsprof.profile(profiled)
            calls.append('after_profiled')
        thread = threading.Thread(target=do_profile)
        bzrlib.lsprof.BzrProfiler.profiler_lock.acquire()
        try:
            try:
                thread.start()
            finally:
                bzrlib.lsprof.BzrProfiler.profiler_lock.release()
        finally:
            thread.join()
        self.assertLength(2, calls)