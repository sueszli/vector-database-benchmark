from __future__ import print_function
from time import time, sleep
import contextlib
import random
import weakref
import gc
import gevent.threadpool
from gevent.threadpool import ThreadPool
import gevent
from gevent.exceptions import InvalidThreadUseError
import gevent.testing as greentest
from gevent.testing import ExpectedException
from gevent.testing import PYPY

@contextlib.contextmanager
def disabled_gc():
    if False:
        return 10
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()

class TestCase(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT
    pool = None
    _all_pools = ()
    ClassUnderTest = ThreadPool

    def _FUT(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ClassUnderTest

    def _makeOne(self, maxsize, create_all_worker_threads=greentest.RUN_LEAKCHECKS):
        if False:
            while True:
                i = 10
        self.pool = pool = self._FUT()(maxsize)
        self._all_pools += (pool,)
        if create_all_worker_threads:
            self.pool.size = maxsize
        return pool

    def cleanup(self):
        if False:
            print('Hello World!')
        self.pool = None
        (all_pools, self._all_pools) = (self._all_pools, ())
        for pool in all_pools:
            kill = getattr(pool, 'kill', None) or getattr(pool, 'shutdown')
            kill()
            del kill
        if greentest.RUN_LEAKCHECKS:
            for _ in range(3):
                gc.collect()

class PoolBasicTests(TestCase):

    def test_execute_async(self):
        if False:
            while True:
                i = 10
        pool = self._makeOne(2)
        r = []
        first = pool.spawn(r.append, 1)
        first.get()
        self.assertEqual(r, [1])
        gevent.sleep(0)
        pool.apply_async(r.append, (2,))
        self.assertEqual(r, [1])
        pool.apply_async(r.append, (3,))
        self.assertEqual(r, [1])
        pool.apply_async(r.append, (4,))
        self.assertEqual(r, [1])
        gevent.sleep(0.01)
        self.assertEqualFlakyRaceCondition(sorted(r), [1, 2, 3, 4])

    def test_apply(self):
        if False:
            while True:
                i = 10
        pool = self._makeOne(1)
        result = pool.apply(lambda a: ('foo', a), (1,))
        self.assertEqual(result, ('foo', 1))

    def test_apply_raises(self):
        if False:
            print('Hello World!')
        pool = self._makeOne(1)

        def raiser():
            if False:
                for i in range(10):
                    print('nop')
            raise ExpectedException()
        with self.assertRaises(ExpectedException):
            pool.apply(raiser)
    test_apply_raises.error_fatal = False

    def test_init_valueerror(self):
        if False:
            i = 10
            return i + 15
        self.switch_expected = False
        with self.assertRaises(ValueError):
            self._makeOne(-1)

class TimingWrapper(object):

    def __init__(self, the_func):
        if False:
            while True:
                i = 10
        self.func = the_func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        t = time()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = time() - t

def sqr(x, wait=0.0):
    if False:
        i = 10
        return i + 15
    sleep(wait)
    return x * x

def sqr_random_sleep(x):
    if False:
        return 10
    sleep(random.random() * 0.1)
    return x * x
(TIMEOUT1, TIMEOUT2, TIMEOUT3) = (0.082, 0.035, 0.14)

class _AbstractPoolTest(TestCase):
    size = 1
    MAP_IS_GEN = False

    def setUp(self):
        if False:
            return 10
        greentest.TestCase.setUp(self)
        self._makeOne(self.size)

    @greentest.ignores_leakcheck
    def test_map(self):
        if False:
            for i in range(10):
                print('nop')
        pmap = self.pool.map
        if self.MAP_IS_GEN:
            pmap = lambda f, i: list(self.pool.map(f, i))
        self.assertEqual(pmap(sqr, range(10)), list(map(sqr, range(10))))
        self.assertEqual(pmap(sqr, range(100)), list(map(sqr, range(100))))
        self.pool.kill()
        del self.pool
        del pmap
SMALL_RANGE = 10
LARGE_RANGE = 1000
if greentest.PYPY and (greentest.WIN or greentest.RUN_COVERAGE) or greentest.RUN_LEAKCHECKS:
    LARGE_RANGE = 50

class TestPool(_AbstractPoolTest):

    def test_greenlet_class(self):
        if False:
            i = 10
            return i + 15
        from greenlet import getcurrent
        from gevent.threadpool import _WorkerGreenlet
        worker_greenlet = self.pool.apply(getcurrent)
        self.assertIsInstance(worker_greenlet, _WorkerGreenlet)
        r = repr(worker_greenlet)
        self.assertIn('ThreadPoolWorker', r)
        self.assertIn('thread_ident', r)
        self.assertIn('hub=', r)
        from gevent.util import format_run_info
        info = '\n'.join(format_run_info())
        self.assertIn('<ThreadPoolWorker', info)

    def test_apply(self):
        if False:
            print('Hello World!')
        papply = self.pool.apply
        self.assertEqual(papply(sqr, (5,)), sqr(5))
        self.assertEqual(papply(sqr, (), {'x': 3}), sqr(x=3))

    def test_async(self):
        if False:
            return 10
        res = self.pool.apply_async(sqr, (7, TIMEOUT1))
        get = TimingWrapper(res.get)
        self.assertEqual(get(), 49)
        self.assertTimeoutAlmostEqual(get.elapsed, TIMEOUT1, 1)

    def test_async_callback(self):
        if False:
            return 10
        result = []
        res = self.pool.apply_async(sqr, (7, TIMEOUT1), callback=result.append)
        get = TimingWrapper(res.get)
        self.assertEqual(get(), 49)
        self.assertTimeoutAlmostEqual(get.elapsed, TIMEOUT1, 1)
        gevent.sleep(0)
        self.assertEqual(result, [49])

    def test_async_timeout(self):
        if False:
            print('Hello World!')
        res = self.pool.apply_async(sqr, (6, TIMEOUT2 + 0.2))
        get = TimingWrapper(res.get)
        self.assertRaises(gevent.Timeout, get, timeout=TIMEOUT2)
        self.assertTimeoutAlmostEqual(get.elapsed, TIMEOUT2, 1)
        self.pool.join()

    def test_imap_list_small(self):
        if False:
            print('Hello World!')
        it = self.pool.imap(sqr, range(SMALL_RANGE))
        self.assertEqual(list(it), list(map(sqr, range(SMALL_RANGE))))

    def test_imap_it_small(self):
        if False:
            i = 10
            return i + 15
        it = self.pool.imap(sqr, range(SMALL_RANGE))
        for i in range(SMALL_RANGE):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, next, it)

    def test_imap_it_large(self):
        if False:
            i = 10
            return i + 15
        it = self.pool.imap(sqr, range(LARGE_RANGE))
        for i in range(LARGE_RANGE):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, next, it)

    def test_imap_gc(self):
        if False:
            while True:
                i = 10
        it = self.pool.imap(sqr, range(SMALL_RANGE))
        for i in range(SMALL_RANGE):
            self.assertEqual(next(it), i * i)
            gc.collect()
        self.assertRaises(StopIteration, next, it)

    def test_imap_unordered_gc(self):
        if False:
            for i in range(10):
                print('nop')
        it = self.pool.imap_unordered(sqr, range(SMALL_RANGE))
        result = []
        for _ in range(SMALL_RANGE):
            result.append(next(it))
            gc.collect()
        with self.assertRaises(StopIteration):
            next(it)
        self.assertEqual(sorted(result), [x * x for x in range(SMALL_RANGE)])

    def test_imap_random(self):
        if False:
            i = 10
            return i + 15
        it = self.pool.imap(sqr_random_sleep, range(SMALL_RANGE))
        self.assertEqual(list(it), list(map(sqr, range(SMALL_RANGE))))

    def test_imap_unordered(self):
        if False:
            while True:
                i = 10
        it = self.pool.imap_unordered(sqr, range(LARGE_RANGE))
        self.assertEqual(sorted(it), list(map(sqr, range(LARGE_RANGE))))
        it = self.pool.imap_unordered(sqr, range(LARGE_RANGE))
        self.assertEqual(sorted(it), list(map(sqr, range(LARGE_RANGE))))

    def test_imap_unordered_random(self):
        if False:
            return 10
        it = self.pool.imap_unordered(sqr_random_sleep, range(SMALL_RANGE))
        self.assertEqual(sorted(it), list(map(sqr, range(SMALL_RANGE))))

    def test_terminate(self):
        if False:
            return 10
        size = self.size or 10
        result = self.pool.map_async(sleep, [0.1] * (size * 2))
        gevent.sleep(0.1)
        try:
            with self.runs_in_given_time(0.1 * self.size + 0.5, min_time=0):
                self.pool.kill()
        finally:
            result.join()

    def sleep(self, x):
        if False:
            return 10
        sleep(float(x) / 10.0)
        return str(x)

    def test_imap_unordered_sleep(self):
        if False:
            for i in range(10):
                print('nop')
        result = list(self.pool.imap_unordered(self.sleep, [10, 1, 2]))
        if self.pool.size == 1:
            expected = ['10', '1', '2']
        else:
            expected = ['1', '2', '10']
        self.assertEqual(result, expected)

class TestPool2(TestPool):
    size = 2

    @greentest.ignores_leakcheck
    def test_recursive_apply(self):
        if False:
            while True:
                i = 10
        p = self.pool

        def a():
            if False:
                for i in range(10):
                    print('nop')
            return p.apply(b)

        def b():
            if False:
                for i in range(10):
                    print('nop')
            gevent.sleep()
            gevent.sleep(0.001)
            return 'B'
        result = p.apply(a)
        self.assertEqual(result, 'B')

@greentest.ignores_leakcheck
class TestPool3(TestPool):
    size = 3

@greentest.ignores_leakcheck
class TestPool10(TestPool):
    size = 10

class TestJoinEmpty(TestCase):
    switch_expected = False

    @greentest.skipIf(greentest.PYPY and greentest.LIBUV and greentest.RUNNING_ON_TRAVIS, 'This sometimes appears to crash in PyPy2 5.9.0, but never crashes on macOS or local Ubunto with same PyPy version')
    def test(self):
        if False:
            i = 10
            return i + 15
        pool = self._makeOne(1)
        pool.join()

class TestSpawn(TestCase):
    switch_expected = True

    @greentest.ignores_leakcheck
    def test_basics(self):
        if False:
            while True:
                i = 10
        pool = self._makeOne(1)
        self.assertEqual(len(pool), 0)
        log = []
        sleep_n_log = lambda item, seconds: [sleep(seconds), log.append(item)]
        pool.spawn(sleep_n_log, 'a', 0.1)
        self.assertEqual(len(pool), 1)
        pool.spawn(sleep_n_log, 'b', 0.1)
        self.assertEqual(len(pool), 2)
        gevent.sleep(0.15)
        self.assertEqual(log, ['a'])
        self.assertEqual(len(pool), 1)
        gevent.sleep(0.15)
        self.assertEqual(log, ['a', 'b'])
        self.assertEqual(len(pool), 0)

    @greentest.ignores_leakcheck
    def test_cannot_spawn_from_other_thread(self):
        if False:
            while True:
                i = 10
        pool1 = self._makeOne(1)
        pool2 = self._makeOne(2)

        def func():
            if False:
                print('Hello World!')
            pool2.spawn(lambda : 'Hi')
        res = pool1.spawn(func)
        with self.assertRaises(InvalidThreadUseError):
            res.get()

def error_iter():
    if False:
        while True:
            i = 10
    yield 1
    yield 2
    raise greentest.ExpectedException

class TestErrorInIterator(TestCase):
    error_fatal = False

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool = self._makeOne(3)
        self.assertRaises(greentest.ExpectedException, self.pool.map, lambda x: None, error_iter())
        gevent.sleep(0.001)

    def test_unordered(self):
        if False:
            print('Hello World!')
        self.pool = self._makeOne(3)

        def unordered():
            if False:
                return 10
            return list(self.pool.imap_unordered(lambda x: None, error_iter()))
        self.assertRaises(greentest.ExpectedException, unordered)
        gevent.sleep(0.001)

class TestMaxsize(TestCase):

    def test_inc(self):
        if False:
            i = 10
            return i + 15
        self.pool = self._makeOne(0)
        done = []
        gevent.spawn(self.pool.spawn, done.append, 1)
        gevent.spawn_later(0.01, self.pool.spawn, done.append, 2)
        gevent.sleep(0.02)
        self.assertEqual(done, [])
        self.pool.maxsize = 1
        gevent.sleep(0.02)
        self.assertEqualFlakyRaceCondition(done, [1, 2])

    @greentest.ignores_leakcheck
    def test_setzero(self):
        if False:
            print('Hello World!')
        pool = self.pool = self._makeOne(3)
        pool.spawn(sleep, 0.1)
        pool.spawn(sleep, 0.2)
        pool.spawn(sleep, 0.3)
        gevent.sleep(0.2)
        self.assertGreaterEqual(pool.size, 2)
        pool.maxsize = 0
        gevent.sleep(0.2)
        self.assertEqualFlakyRaceCondition(pool.size, 0)

class TestSize(TestCase):

    @greentest.reraises_flaky_race_condition()
    def test(self):
        if False:
            i = 10
            return i + 15
        pool = self.pool = self._makeOne(2, create_all_worker_threads=False)
        self.assertEqual(pool.size, 0)
        pool.size = 1
        self.assertEqual(pool.size, 1)
        pool.size = 2
        self.assertEqual(pool.size, 2)
        pool.size = 1
        self.assertEqual(pool.size, 1)
        with self.assertRaises(ValueError):
            pool.size = -1
        with self.assertRaises(ValueError):
            pool.size = 3
        pool.size = 0
        self.assertEqual(pool.size, 0)
        pool.size = 2
        self.assertEqual(pool.size, 2)

class TestRef(TestCase):

    def test(self):
        if False:
            print('Hello World!')
        pool = self.pool = self._makeOne(2)
        refs = []
        obj = SomeClass()
        obj.refs = refs
        func = obj.func
        del obj
        with disabled_gc():
            result = pool.apply(func, (Object(),), {'kwarg1': Object()})
            self.assertIsInstance(result, Object)
            gevent.sleep(0.1)
            refs.append(weakref.ref(func))
            del func, result
            if PYPY:
                gc.collect()
                gc.collect()
            for r in refs:
                self.assertIsNone(r())
            self.assertEqual(4, len(refs))

class Object(object):
    pass

class SomeClass(object):
    refs = None

    def func(self, arg1, kwarg1=None):
        if False:
            i = 10
            return i + 15
        result = Object()
        self.refs.extend([weakref.ref(x) for x in (arg1, kwarg1, result)])
        return result

def noop():
    if False:
        for i in range(10):
            print('nop')
    pass

class TestRefCount(TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        pool = self._makeOne(1)
        pool.spawn(noop)
        gevent.sleep(0)
        pool.kill()
from gevent import monkey

@greentest.skipUnless(hasattr(gevent.threadpool, 'ThreadPoolExecutor'), 'Requires ThreadPoolExecutor')
class TestTPE(_AbstractPoolTest):
    size = 1
    MAP_IS_GEN = True

    @property
    def ClassUnderTest(self):
        if False:
            for i in range(10):
                print('nop')
        return gevent.threadpool.ThreadPoolExecutor
    MONKEY_PATCHED = False

    @property
    def FutureTimeoutError(self):
        if False:
            for i in range(10):
                print('nop')
        from concurrent.futures import TimeoutError as FutureTimeoutError
        return FutureTimeoutError

    @property
    def cf_wait(self):
        if False:
            print('Hello World!')
        from concurrent.futures import wait as cf_wait
        return cf_wait

    @property
    def cf_as_completed(self):
        if False:
            return 10
        from concurrent.futures import as_completed as cf_as_completed
        return cf_as_completed

    @greentest.ignores_leakcheck
    def test_future(self):
        if False:
            while True:
                i = 10
        self.assertEqual(monkey.is_module_patched('threading'), self.MONKEY_PATCHED)
        pool = self.pool
        calledback = []

        def fn():
            if False:
                print('Hello World!')
            gevent.sleep(0.5)
            return 42

        def callback(future):
            if False:
                return 10
            future.calledback += 1
            raise greentest.ExpectedException('Expected, ignored')
        future = pool.submit(fn)
        future.calledback = 0
        future.add_done_callback(callback)
        self.assertRaises(self.FutureTimeoutError, future.result, timeout=0.001)

        def spawned():
            if False:
                i = 10
                return i + 15
            return 2016
        spawned_greenlet = gevent.spawn(spawned)
        self.assertEqual(future.result(), 42)
        self.assertTrue(future.done())
        self.assertFalse(future.cancelled())
        gevent.sleep()
        self.assertEqual(future.calledback, 1)
        self.assertTrue(spawned_greenlet.ready())
        self.assertEqual(spawned_greenlet.value, 2016)
        future.add_done_callback(lambda f: calledback.append(True))
        self.assertEqual(calledback, [True])
        (done, _not_done) = self.cf_wait((future,))
        self.assertEqual(list(done), [future])
        self.assertEqual(list(self.cf_as_completed((future,))), [future])
        self.assertEqual(future.calledback, 1)
        gevent.sleep()
        self.assertEqual(future.calledback, 1)
        pool.kill()
        del future
        del pool
        del self.pool

    @greentest.ignores_leakcheck
    def test_future_wait_module_function(self):
        if False:
            print('Hello World!')
        self.assertEqual(monkey.is_module_patched('threading'), self.MONKEY_PATCHED)
        pool = self.pool

        def fn():
            if False:
                i = 10
                return i + 15
            gevent.sleep(0.5)
            return 42
        future = pool.submit(fn)
        if self.MONKEY_PATCHED:
            (_done, not_done) = self.cf_wait((future,), timeout=0.001)
            self.assertEqual(list(not_done), [future])

            def spawned():
                if False:
                    while True:
                        i = 10
                return 2016
            spawned_greenlet = gevent.spawn(spawned)
            (done, _not_done) = self.cf_wait((future,))
            self.assertEqual(list(done), [future])
            self.assertTrue(spawned_greenlet.ready())
            self.assertEqual(spawned_greenlet.value, 2016)
        else:
            self.assertRaises(AttributeError, self.cf_wait, (future,))
        pool.kill()
        del future
        del pool
        del self.pool

    @greentest.ignores_leakcheck
    def test_future_wait_gevent_function(self):
        if False:
            print('Hello World!')
        self.assertEqual(monkey.is_module_patched('threading'), self.MONKEY_PATCHED)
        pool = self.pool

        def fn():
            if False:
                i = 10
                return i + 15
            gevent.sleep(0.5)
            return 42
        future = pool.submit(fn)

        def spawned():
            if False:
                while True:
                    i = 10
            return 2016
        spawned_greenlet = gevent.spawn(spawned)
        done = gevent.wait((future,))
        self.assertEqual(list(done), [future])
        self.assertTrue(spawned_greenlet.ready())
        self.assertEqual(spawned_greenlet.value, 2016)
        pool.kill()
        del future
        del pool
        del self.pool

class TestThreadResult(greentest.TestCase):

    def test_exception_in_on_async_doesnt_crash(self):
        if False:
            for i in range(10):
                print('nop')
        called = []

        class MyException(Exception):
            pass

        def bad_when_ready():
            if False:
                i = 10
                return i + 15
            called.append(1)
            raise MyException
        tr = gevent.threadpool.ThreadResult(None, gevent.get_hub(), bad_when_ready)

        def wake():
            if False:
                for i in range(10):
                    print('nop')
            called.append(1)
            tr.set(42)
        gevent.spawn(wake).get()
        with self.assertRaises(MyException):
            for _ in range(5):
                gevent.sleep(0.001)
        self.assertEqual(called, [1, 1])
        self.assertIsNone(tr.value)
        self.assertIsNotNone(tr.receiver)

class TestWorkerProfileAndTrace(TestCase):
    old_profile = None
    old_trace = None

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestWorkerProfileAndTrace, self).setUp()
        self.old_profile = gevent.threadpool._get_thread_profile()
        self.old_trace = gevent.threadpool._get_thread_trace()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        import threading
        threading.setprofile(self.old_profile)
        threading.settrace(self.old_trace)
        super(TestWorkerProfileAndTrace, self).tearDown()

    def test_get_profile(self):
        if False:
            print('Hello World!')
        import threading
        threading.setprofile(self)
        self.assertIs(gevent.threadpool._get_thread_profile(), self)

    def test_get_trace(self):
        if False:
            for i in range(10):
                print('nop')
        import threading
        threading.settrace(self)
        self.assertIs(gevent.threadpool._get_thread_trace(), self)

    def _test_func_called_in_task(self, func):
        if False:
            for i in range(10):
                print('nop')
        import threading
        import sys
        setter = getattr(threading, 'set' + func)
        getter = getattr(sys, 'get' + func)
        called = [0]

        def callback(*_args):
            if False:
                print('Hello World!')
            called[0] += 1

        def task():
            if False:
                i = 10
                return i + 15
            test.assertIsNotNone(getter)
            return 1701
        before_task = []
        after_task = []
        test = self

        class Pool(ThreadPool):

            class _WorkerGreenlet(ThreadPool._WorkerGreenlet):

                def _before_run_task(self, func, *args):
                    if False:
                        print('Hello World!')
                    before_task.append(func)
                    before_task.append(getter())
                    ThreadPool._WorkerGreenlet._before_run_task(self, func, *args)
                    before_task.append(getter())

                def _after_run_task(self, func, *args):
                    if False:
                        print('Hello World!')
                    after_task.append(func)
                    after_task.append(getter())
                    ThreadPool._WorkerGreenlet._after_run_task(self, func, *args)
                    after_task.append(getter())
        self.ClassUnderTest = Pool
        pool = self._makeOne(1, create_all_worker_threads=True)
        assert isinstance(pool, Pool)
        setter(callback)
        res = pool.apply(task)
        self.assertEqual(res, 1701)
        self.assertGreaterEqual(called[0], 1)
        pool.kill()
        self.assertEqual(before_task, [task, None, callback])
        self.assertEqual(after_task, [task, callback, None])

    def test_profile_called_in_task(self):
        if False:
            return 10
        self._test_func_called_in_task('profile')

    def test_trace_called_in_task(self):
        if False:
            while True:
                i = 10
        self._test_func_called_in_task('trace')
if __name__ == '__main__':
    greentest.main()