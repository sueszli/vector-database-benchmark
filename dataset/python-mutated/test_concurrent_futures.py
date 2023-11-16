from test import support
from test.support import import_helper
from test.support import threading_helper
import_helper.import_module('_multiprocessing')
from test.support import hashlib_helper
from test.support.script_helper import assert_python_ok
import contextlib
import itertools
import logging
from logging.handlers import QueueHandler
import os
import queue
import sys
import threading
import time
import unittest
import weakref
from pickle import PicklingError
from concurrent import futures
from concurrent.futures._base import PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED, Future, BrokenExecutor
from concurrent.futures.process import BrokenProcessPool, _check_system_limits
import multiprocessing.process
import multiprocessing.util
import multiprocessing as mp
if support.check_sanitizer(address=True, memory=True):
    raise unittest.SkipTest('test too slow on ASAN/MSAN build')

def create_future(state=PENDING, exception=None, result=None):
    if False:
        print('Hello World!')
    f = Future()
    f._state = state
    f._exception = exception
    f._result = result
    return f
PENDING_FUTURE = create_future(state=PENDING)
RUNNING_FUTURE = create_future(state=RUNNING)
CANCELLED_FUTURE = create_future(state=CANCELLED)
CANCELLED_AND_NOTIFIED_FUTURE = create_future(state=CANCELLED_AND_NOTIFIED)
EXCEPTION_FUTURE = create_future(state=FINISHED, exception=OSError())
SUCCESSFUL_FUTURE = create_future(state=FINISHED, result=42)
INITIALIZER_STATUS = 'uninitialized'

def mul(x, y):
    if False:
        print('Hello World!')
    return x * y

def capture(*args, **kwargs):
    if False:
        while True:
            i = 10
    return (args, kwargs)

def sleep_and_raise(t):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(t)
    raise Exception('this is an exception')

def sleep_and_print(t, msg):
    if False:
        return 10
    time.sleep(t)
    print(msg)
    sys.stdout.flush()

def init(x):
    if False:
        i = 10
        return i + 15
    global INITIALIZER_STATUS
    INITIALIZER_STATUS = x

def get_init_status():
    if False:
        print('Hello World!')
    return INITIALIZER_STATUS

def init_fail(log_queue=None):
    if False:
        print('Hello World!')
    if log_queue is not None:
        logger = logging.getLogger('concurrent.futures')
        logger.addHandler(QueueHandler(log_queue))
        logger.setLevel('CRITICAL')
        logger.propagate = False
    time.sleep(0.1)
    raise ValueError('error in initializer')

class MyObject(object):

    def my_method(self):
        if False:
            i = 10
            return i + 15
        pass

class EventfulGCObj:

    def __init__(self, mgr):
        if False:
            i = 10
            return i + 15
        self.event = mgr.Event()

    def __del__(self):
        if False:
            print('Hello World!')
        self.event.set()

def make_dummy_object(_):
    if False:
        return 10
    return MyObject()

class BaseTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._thread_key = threading_helper.threading_setup()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        support.reap_children()
        threading_helper.threading_cleanup(*self._thread_key)

class ExecutorMixin:
    worker_count = 5
    executor_kwargs = {}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.t1 = time.monotonic()
        if hasattr(self, 'ctx'):
            self.executor = self.executor_type(max_workers=self.worker_count, mp_context=self.get_context(), **self.executor_kwargs)
        else:
            self.executor = self.executor_type(max_workers=self.worker_count, **self.executor_kwargs)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.executor.shutdown(wait=True)
        self.executor = None
        dt = time.monotonic() - self.t1
        if support.verbose:
            print('%.2fs' % dt, end=' ')
        self.assertLess(dt, 300, 'synchronization issue: test lasted too long')
        super().tearDown()

    def get_context(self):
        if False:
            for i in range(10):
                print('nop')
        return mp.get_context(self.ctx)

class ThreadPoolMixin(ExecutorMixin):
    executor_type = futures.ThreadPoolExecutor

class ProcessPoolForkMixin(ExecutorMixin):
    executor_type = futures.ProcessPoolExecutor
    ctx = 'fork'

    def get_context(self):
        if False:
            return 10
        try:
            _check_system_limits()
        except NotImplementedError:
            self.skipTest('ProcessPoolExecutor unavailable on this system')
        if sys.platform == 'win32':
            self.skipTest('require unix system')
        return super().get_context()

class ProcessPoolSpawnMixin(ExecutorMixin):
    executor_type = futures.ProcessPoolExecutor
    ctx = 'spawn'

    def get_context(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            _check_system_limits()
        except NotImplementedError:
            self.skipTest('ProcessPoolExecutor unavailable on this system')
        return super().get_context()

class ProcessPoolForkserverMixin(ExecutorMixin):
    executor_type = futures.ProcessPoolExecutor
    ctx = 'forkserver'

    def get_context(self):
        if False:
            print('Hello World!')
        try:
            _check_system_limits()
        except NotImplementedError:
            self.skipTest('ProcessPoolExecutor unavailable on this system')
        if sys.platform == 'win32':
            self.skipTest('require unix system')
        return super().get_context()

def create_executor_tests(mixin, bases=(BaseTestCase,), executor_mixins=(ThreadPoolMixin, ProcessPoolForkMixin, ProcessPoolForkserverMixin, ProcessPoolSpawnMixin)):
    if False:
        i = 10
        return i + 15

    def strip_mixin(name):
        if False:
            while True:
                i = 10
        if name.endswith(('Mixin', 'Tests')):
            return name[:-5]
        elif name.endswith('Test'):
            return name[:-4]
        else:
            return name
    for exe in executor_mixins:
        name = '%s%sTest' % (strip_mixin(exe.__name__), strip_mixin(mixin.__name__))
        cls = type(name, (mixin,) + (exe,) + bases, {})
        globals()[name] = cls

class InitializerMixin(ExecutorMixin):
    worker_count = 2

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        global INITIALIZER_STATUS
        INITIALIZER_STATUS = 'uninitialized'
        self.executor_kwargs = dict(initializer=init, initargs=('initialized',))
        super().setUp()

    def test_initializer(self):
        if False:
            return 10
        futures = [self.executor.submit(get_init_status) for _ in range(self.worker_count)]
        for f in futures:
            self.assertEqual(f.result(), 'initialized')

class FailingInitializerMixin(ExecutorMixin):
    worker_count = 2

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'ctx'):
            self.mp_context = self.get_context()
            self.log_queue = self.mp_context.Queue()
            self.executor_kwargs = dict(initializer=init_fail, initargs=(self.log_queue,))
        else:
            self.mp_context = None
            self.log_queue = None
            self.executor_kwargs = dict(initializer=init_fail)
        super().setUp()

    def test_initializer(self):
        if False:
            print('Hello World!')
        with self._assert_logged('ValueError: error in initializer'):
            try:
                future = self.executor.submit(get_init_status)
            except BrokenExecutor:
                pass
            else:
                with self.assertRaises(BrokenExecutor):
                    future.result()
            t1 = time.monotonic()
            while not self.executor._broken:
                if time.monotonic() - t1 > 5:
                    self.fail('executor not broken after 5 s.')
                time.sleep(0.01)
            with self.assertRaises(BrokenExecutor):
                self.executor.submit(get_init_status)

    @contextlib.contextmanager
    def _assert_logged(self, msg):
        if False:
            return 10
        if self.log_queue is not None:
            yield
            output = []
            try:
                while True:
                    output.append(self.log_queue.get_nowait().getMessage())
            except queue.Empty:
                pass
        else:
            with self.assertLogs('concurrent.futures', 'CRITICAL') as cm:
                yield
            output = cm.output
        self.assertTrue(any((msg in line for line in output)), output)
create_executor_tests(InitializerMixin)
create_executor_tests(FailingInitializerMixin)

class ExecutorShutdownTest:

    def test_run_after_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.executor.shutdown()
        self.assertRaises(RuntimeError, self.executor.submit, pow, 2, 5)

    def test_interpreter_shutdown(self):
        if False:
            print('Hello World!')
        (rc, out, err) = assert_python_ok('-c', 'if 1:\n            from concurrent.futures import {executor_type}\n            from time import sleep\n            from test.test_concurrent_futures import sleep_and_print\n            if __name__ == "__main__":\n                context = \'{context}\'\n                if context == "":\n                    t = {executor_type}(5)\n                else:\n                    from multiprocessing import get_context\n                    context = get_context(context)\n                    t = {executor_type}(5, mp_context=context)\n                t.submit(sleep_and_print, 1.0, "apple")\n            '.format(executor_type=self.executor_type.__name__, context=getattr(self, 'ctx', '')))
        self.assertFalse(err)
        self.assertEqual(out.strip(), b'apple')

    def test_submit_after_interpreter_shutdown(self):
        if False:
            return 10
        (rc, out, err) = assert_python_ok('-c', 'if 1:\n            import atexit\n            @atexit.register\n            def run_last():\n                try:\n                    t.submit(id, None)\n                except RuntimeError:\n                    print("runtime-error")\n                    raise\n            from concurrent.futures import {executor_type}\n            if __name__ == "__main__":\n                context = \'{context}\'\n                if not context:\n                    t = {executor_type}(5)\n                else:\n                    from multiprocessing import get_context\n                    context = get_context(context)\n                    t = {executor_type}(5, mp_context=context)\n                    t.submit(id, 42).result()\n            '.format(executor_type=self.executor_type.__name__, context=getattr(self, 'ctx', '')))
        self.assertIn('RuntimeError: cannot schedule new futures', err.decode())
        self.assertEqual(out.strip(), b'runtime-error')

    def test_hang_issue12364(self):
        if False:
            return 10
        fs = [self.executor.submit(time.sleep, 0.1) for _ in range(50)]
        self.executor.shutdown()
        for f in fs:
            f.result()

    def test_cancel_futures(self):
        if False:
            while True:
                i = 10
        assert self.worker_count <= 5, 'test needs few workers'
        fs = [self.executor.submit(time.sleep, 0.1) for _ in range(50)]
        self.executor.shutdown(cancel_futures=True)
        cancelled = [fut for fut in fs if fut.cancelled()]
        self.assertGreater(len(cancelled), 20)
        others = [fut for fut in fs if not fut.cancelled()]
        for fut in others:
            self.assertTrue(fut.done(), msg=f'fut._state={fut._state!r}')
            self.assertIsNone(fut.exception())
        self.assertGreater(len(others), 0)

    def test_hang_gh83386(self):
        if False:
            return 10
        "shutdown(wait=False) doesn't hang at exit with running futures.\n\n        See https://github.com/python/cpython/issues/83386.\n        "
        if self.executor_type == futures.ProcessPoolExecutor:
            raise unittest.SkipTest('Hangs, see https://github.com/python/cpython/issues/83386')
        (rc, out, err) = assert_python_ok('-c', 'if True:\n            from concurrent.futures import {executor_type}\n            from test.test_concurrent_futures import sleep_and_print\n            if __name__ == "__main__":\n                if {context!r}: multiprocessing.set_start_method({context!r})\n                t = {executor_type}(max_workers=3)\n                t.submit(sleep_and_print, 1.0, "apple")\n                t.shutdown(wait=False)\n            '.format(executor_type=self.executor_type.__name__, context=getattr(self, 'ctx', None)))
        self.assertFalse(err)
        self.assertEqual(out.strip(), b'apple')

class ThreadPoolShutdownTest(ThreadPoolMixin, ExecutorShutdownTest, BaseTestCase):

    def test_threads_terminate(self):
        if False:
            return 10

        def acquire_lock(lock):
            if False:
                return 10
            lock.acquire()
        sem = threading.Semaphore(0)
        for i in range(3):
            self.executor.submit(acquire_lock, sem)
        self.assertEqual(len(self.executor._threads), 3)
        for i in range(3):
            sem.release()
        self.executor.shutdown()
        for t in self.executor._threads:
            t.join()

    def test_context_manager_shutdown(self):
        if False:
            i = 10
            return i + 15
        with futures.ThreadPoolExecutor(max_workers=5) as e:
            executor = e
            self.assertEqual(list(e.map(abs, range(-5, 5))), [5, 4, 3, 2, 1, 0, 1, 2, 3, 4])
        for t in executor._threads:
            t.join()

    def test_del_shutdown(self):
        if False:
            i = 10
            return i + 15
        executor = futures.ThreadPoolExecutor(max_workers=5)
        res = executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor
        for t in threads:
            t.join()
        assert all([r == abs(v) for (r, v) in zip(res, range(-5, 5))])

    def test_shutdown_no_wait(self):
        if False:
            print('Hello World!')
        executor = futures.ThreadPoolExecutor(max_workers=5)
        res = executor.map(abs, range(-5, 5))
        threads = executor._threads
        executor.shutdown(wait=False)
        for t in threads:
            t.join()
        assert all([r == abs(v) for (r, v) in zip(res, range(-5, 5))])

    def test_thread_names_assigned(self):
        if False:
            while True:
                i = 10
        executor = futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix='SpecialPool')
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor
        support.gc_collect()
        for t in threads:
            self.assertRegex(t.name, '^SpecialPool_[0-4]$')
            t.join()

    def test_thread_names_default(self):
        if False:
            while True:
                i = 10
        executor = futures.ThreadPoolExecutor(max_workers=5)
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor
        support.gc_collect()
        for t in threads:
            self.assertRegex(t.name, 'ThreadPoolExecutor-\\d+_[0-4]$')
            t.join()

    def test_cancel_futures_wait_false(self):
        if False:
            while True:
                i = 10
        (rc, out, err) = assert_python_ok('-c', 'if True:\n            from concurrent.futures import ThreadPoolExecutor\n            from test.test_concurrent_futures import sleep_and_print\n            if __name__ == "__main__":\n                t = ThreadPoolExecutor()\n                t.submit(sleep_and_print, .1, "apple")\n                t.shutdown(wait=False, cancel_futures=True)\n            '.format(executor_type=self.executor_type.__name__))
        self.assertFalse(err)
        self.assertEqual(out.strip(), b'apple')

class ProcessPoolShutdownTest(ExecutorShutdownTest):

    def test_processes_terminate(self):
        if False:
            i = 10
            return i + 15

        def acquire_lock(lock):
            if False:
                return 10
            lock.acquire()
        mp_context = self.get_context()
        if mp_context.get_start_method(allow_none=False) == 'fork':
            expected_num_processes = self.worker_count
        else:
            expected_num_processes = 3
        sem = mp_context.Semaphore(0)
        for _ in range(3):
            self.executor.submit(acquire_lock, sem)
        self.assertEqual(len(self.executor._processes), expected_num_processes)
        for _ in range(3):
            sem.release()
        processes = self.executor._processes
        self.executor.shutdown()
        for p in processes.values():
            p.join()

    def test_context_manager_shutdown(self):
        if False:
            return 10
        with futures.ProcessPoolExecutor(max_workers=5, mp_context=self.get_context()) as e:
            processes = e._processes
            self.assertEqual(list(e.map(abs, range(-5, 5))), [5, 4, 3, 2, 1, 0, 1, 2, 3, 4])
        for p in processes.values():
            p.join()

    def test_del_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        executor = futures.ProcessPoolExecutor(max_workers=5, mp_context=self.get_context())
        res = executor.map(abs, range(-5, 5))
        executor_manager_thread = executor._executor_manager_thread
        processes = executor._processes
        call_queue = executor._call_queue
        executor_manager_thread = executor._executor_manager_thread
        del executor
        support.gc_collect()
        executor_manager_thread.join()
        for p in processes.values():
            p.join()
        call_queue.join_thread()
        assert all([r == abs(v) for (r, v) in zip(res, range(-5, 5))])

    def test_shutdown_no_wait(self):
        if False:
            return 10
        executor = futures.ProcessPoolExecutor(max_workers=5, mp_context=self.get_context())
        res = executor.map(abs, range(-5, 5))
        processes = executor._processes
        call_queue = executor._call_queue
        executor_manager_thread = executor._executor_manager_thread
        executor.shutdown(wait=False)
        executor_manager_thread.join()
        for p in processes.values():
            p.join()
        call_queue.join_thread()
        assert all([r == abs(v) for (r, v) in zip(res, range(-5, 5))])
create_executor_tests(ProcessPoolShutdownTest, executor_mixins=(ProcessPoolForkMixin, ProcessPoolForkserverMixin, ProcessPoolSpawnMixin))

class WaitTests:

    def test_20369(self):
        if False:
            return 10
        future = self.executor.submit(time.sleep, 1.5)
        (done, not_done) = futures.wait([future, future], return_when=futures.ALL_COMPLETED)
        self.assertEqual({future}, done)
        self.assertEqual(set(), not_done)

    def test_first_completed(self):
        if False:
            while True:
                i = 10
        future1 = self.executor.submit(mul, 21, 2)
        future2 = self.executor.submit(time.sleep, 1.5)
        (done, not_done) = futures.wait([CANCELLED_FUTURE, future1, future2], return_when=futures.FIRST_COMPLETED)
        self.assertEqual(set([future1]), done)
        self.assertEqual(set([CANCELLED_FUTURE, future2]), not_done)

    def test_first_completed_some_already_completed(self):
        if False:
            for i in range(10):
                print('nop')
        future1 = self.executor.submit(time.sleep, 1.5)
        (finished, pending) = futures.wait([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE, future1], return_when=futures.FIRST_COMPLETED)
        self.assertEqual(set([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE]), finished)
        self.assertEqual(set([future1]), pending)

    def test_first_exception(self):
        if False:
            while True:
                i = 10
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(sleep_and_raise, 1.5)
        future3 = self.executor.submit(time.sleep, 3)
        (finished, pending) = futures.wait([future1, future2, future3], return_when=futures.FIRST_EXCEPTION)
        self.assertEqual(set([future1, future2]), finished)
        self.assertEqual(set([future3]), pending)

    def test_first_exception_some_already_complete(self):
        if False:
            i = 10
            return i + 15
        future1 = self.executor.submit(divmod, 21, 0)
        future2 = self.executor.submit(time.sleep, 1.5)
        (finished, pending) = futures.wait([SUCCESSFUL_FUTURE, CANCELLED_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE, future1, future2], return_when=futures.FIRST_EXCEPTION)
        self.assertEqual(set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE, future1]), finished)
        self.assertEqual(set([CANCELLED_FUTURE, future2]), pending)

    def test_first_exception_one_already_failed(self):
        if False:
            for i in range(10):
                print('nop')
        future1 = self.executor.submit(time.sleep, 2)
        (finished, pending) = futures.wait([EXCEPTION_FUTURE, future1], return_when=futures.FIRST_EXCEPTION)
        self.assertEqual(set([EXCEPTION_FUTURE]), finished)
        self.assertEqual(set([future1]), pending)

    def test_all_completed(self):
        if False:
            return 10
        future1 = self.executor.submit(divmod, 2, 0)
        future2 = self.executor.submit(mul, 2, 21)
        (finished, pending) = futures.wait([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, future1, future2], return_when=futures.ALL_COMPLETED)
        self.assertEqual(set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, future1, future2]), finished)
        self.assertEqual(set(), pending)

    def test_timeout(self):
        if False:
            return 10
        future1 = self.executor.submit(mul, 6, 7)
        future2 = self.executor.submit(time.sleep, 6)
        (finished, pending) = futures.wait([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE, future1, future2], timeout=5, return_when=futures.ALL_COMPLETED)
        self.assertEqual(set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE, future1]), finished)
        self.assertEqual(set([future2]), pending)

class ThreadPoolWaitTests(ThreadPoolMixin, WaitTests, BaseTestCase):

    def test_pending_calls_race(self):
        if False:
            return 10
        event = threading.Event()

        def future_func():
            if False:
                return 10
            event.wait()
        oldswitchinterval = sys.getswitchinterval()
        sys.setswitchinterval(1e-06)
        try:
            fs = {self.executor.submit(future_func) for i in range(100)}
            event.set()
            futures.wait(fs, return_when=futures.ALL_COMPLETED)
        finally:
            sys.setswitchinterval(oldswitchinterval)
create_executor_tests(WaitTests, executor_mixins=(ProcessPoolForkMixin, ProcessPoolForkserverMixin, ProcessPoolSpawnMixin))

class AsCompletedTests:

    def test_no_timeout(self):
        if False:
            return 10
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(mul, 7, 6)
        completed = set(futures.as_completed([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE, future1, future2]))
        self.assertEqual(set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE, future1, future2]), completed)

    def test_zero_timeout(self):
        if False:
            i = 10
            return i + 15
        future1 = self.executor.submit(time.sleep, 2)
        completed_futures = set()
        try:
            for future in futures.as_completed([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE, future1], timeout=0):
                completed_futures.add(future)
        except futures.TimeoutError:
            pass
        self.assertEqual(set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE, SUCCESSFUL_FUTURE]), completed_futures)

    def test_duplicate_futures(self):
        if False:
            for i in range(10):
                print('nop')
        future1 = self.executor.submit(time.sleep, 2)
        completed = [f for f in futures.as_completed(itertools.repeat(future1, 3))]
        self.assertEqual(len(completed), 1)

    def test_free_reference_yielded_future(self):
        if False:
            for i in range(10):
                print('nop')
        futures_list = [Future() for _ in range(8)]
        futures_list.append(create_future(state=CANCELLED_AND_NOTIFIED))
        futures_list.append(create_future(state=FINISHED, result=42))
        with self.assertRaises(futures.TimeoutError):
            for future in futures.as_completed(futures_list, timeout=0):
                futures_list.remove(future)
                wr = weakref.ref(future)
                del future
                support.gc_collect()
                self.assertIsNone(wr())
        futures_list[0].set_result('test')
        for future in futures.as_completed(futures_list):
            futures_list.remove(future)
            wr = weakref.ref(future)
            del future
            support.gc_collect()
            self.assertIsNone(wr())
            if futures_list:
                futures_list[0].set_result('test')

    def test_correct_timeout_exception_msg(self):
        if False:
            print('Hello World!')
        futures_list = [CANCELLED_AND_NOTIFIED_FUTURE, PENDING_FUTURE, RUNNING_FUTURE, SUCCESSFUL_FUTURE]
        with self.assertRaises(futures.TimeoutError) as cm:
            list(futures.as_completed(futures_list, timeout=0))
        self.assertEqual(str(cm.exception), '2 (of 4) futures unfinished')
create_executor_tests(AsCompletedTests)

class ExecutorTest:

    def test_submit(self):
        if False:
            i = 10
            return i + 15
        future = self.executor.submit(pow, 2, 8)
        self.assertEqual(256, future.result())

    def test_submit_keyword(self):
        if False:
            print('Hello World!')
        future = self.executor.submit(mul, 2, y=8)
        self.assertEqual(16, future.result())
        future = self.executor.submit(capture, 1, self=2, fn=3)
        self.assertEqual(future.result(), ((1,), {'self': 2, 'fn': 3}))
        with self.assertRaises(TypeError):
            self.executor.submit(fn=capture, arg=1)
        with self.assertRaises(TypeError):
            self.executor.submit(arg=1)

    def test_map(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(list(self.executor.map(pow, range(10), range(10))), list(map(pow, range(10), range(10))))
        self.assertEqual(list(self.executor.map(pow, range(10), range(10), chunksize=3)), list(map(pow, range(10), range(10))))

    def test_map_exception(self):
        if False:
            print('Hello World!')
        i = self.executor.map(divmod, [1, 1, 1, 1], [2, 3, 0, 5])
        self.assertEqual(i.__next__(), (0, 1))
        self.assertEqual(i.__next__(), (0, 1))
        self.assertRaises(ZeroDivisionError, i.__next__)

    def test_map_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        results = []
        try:
            for i in self.executor.map(time.sleep, [0, 0, 6], timeout=5):
                results.append(i)
        except futures.TimeoutError:
            pass
        else:
            self.fail('expected TimeoutError')
        self.assertEqual([None, None], results)

    def test_shutdown_race_issue12456(self):
        if False:
            while True:
                i = 10
        self.executor.map(str, [2] * (self.worker_count + 1))
        self.executor.shutdown()

    @support.cpython_only
    def test_no_stale_references(self):
        if False:
            for i in range(10):
                print('nop')
        my_object = MyObject()
        my_object_collected = threading.Event()
        my_object_callback = weakref.ref(my_object, lambda obj: my_object_collected.set())
        self.executor.submit(my_object.my_method)
        del my_object
        collected = my_object_collected.wait(timeout=support.SHORT_TIMEOUT)
        self.assertTrue(collected, 'Stale reference not collected within timeout.')

    def test_max_workers_negative(self):
        if False:
            for i in range(10):
                print('nop')
        for number in (0, -1):
            with self.assertRaisesRegex(ValueError, 'max_workers must be greater than 0'):
                self.executor_type(max_workers=number)

    def test_free_reference(self):
        if False:
            return 10
        for obj in self.executor.map(make_dummy_object, range(10)):
            wr = weakref.ref(obj)
            del obj
            support.gc_collect()
            self.assertIsNone(wr())

class ThreadPoolExecutorTest(ThreadPoolMixin, ExecutorTest, BaseTestCase):

    def test_map_submits_without_iteration(self):
        if False:
            i = 10
            return i + 15
        'Tests verifying issue 11777.'
        finished = []

        def record_finished(n):
            if False:
                return 10
            finished.append(n)
        self.executor.map(record_finished, range(10))
        self.executor.shutdown(wait=True)
        self.assertCountEqual(finished, range(10))

    def test_default_workers(self):
        if False:
            i = 10
            return i + 15
        executor = self.executor_type()
        expected = min(32, (os.cpu_count() or 1) + 4)
        self.assertEqual(executor._max_workers, expected)

    def test_saturation(self):
        if False:
            i = 10
            return i + 15
        executor = self.executor_type(4)

        def acquire_lock(lock):
            if False:
                i = 10
                return i + 15
            lock.acquire()
        sem = threading.Semaphore(0)
        for i in range(15 * executor._max_workers):
            executor.submit(acquire_lock, sem)
        self.assertEqual(len(executor._threads), executor._max_workers)
        for i in range(15 * executor._max_workers):
            sem.release()
        executor.shutdown(wait=True)

    def test_idle_thread_reuse(self):
        if False:
            print('Hello World!')
        executor = self.executor_type()
        executor.submit(mul, 21, 2).result()
        executor.submit(mul, 6, 7).result()
        executor.submit(mul, 3, 14).result()
        self.assertEqual(len(executor._threads), 1)
        executor.shutdown(wait=True)

    @unittest.skipUnless(hasattr(os, 'register_at_fork'), 'need os.register_at_fork')
    def test_hang_global_shutdown_lock(self):
        if False:
            return 10

        def submit(pool):
            if False:
                print('Hello World!')
            pool.submit(submit, pool)
        with futures.ThreadPoolExecutor(1) as pool:
            pool.submit(submit, pool)
            for _ in range(50):
                with futures.ProcessPoolExecutor(1, mp_context=mp.get_context('fork')) as workers:
                    workers.submit(tuple)

class ProcessPoolExecutorTest(ExecutorTest):

    @unittest.skipUnless(sys.platform == 'win32', 'Windows-only process limit')
    def test_max_workers_too_large(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'max_workers must be <= 61'):
            futures.ProcessPoolExecutor(max_workers=62)

    def test_killed_child(self):
        if False:
            while True:
                i = 10
        futures = [self.executor.submit(time.sleep, 3)]
        p = next(iter(self.executor._processes.values()))
        p.terminate()
        for fut in futures:
            self.assertRaises(BrokenProcessPool, fut.result)
        self.assertRaises(BrokenProcessPool, self.executor.submit, pow, 2, 8)

    def test_map_chunksize(self):
        if False:
            return 10

        def bad_map():
            if False:
                i = 10
                return i + 15
            list(self.executor.map(pow, range(40), range(40), chunksize=-1))
        ref = list(map(pow, range(40), range(40)))
        self.assertEqual(list(self.executor.map(pow, range(40), range(40), chunksize=6)), ref)
        self.assertEqual(list(self.executor.map(pow, range(40), range(40), chunksize=50)), ref)
        self.assertEqual(list(self.executor.map(pow, range(40), range(40), chunksize=40)), ref)
        self.assertRaises(ValueError, bad_map)

    @classmethod
    def _test_traceback(cls):
        if False:
            return 10
        raise RuntimeError(123)

    def test_traceback(self):
        if False:
            return 10
        future = self.executor.submit(self._test_traceback)
        with self.assertRaises(Exception) as cm:
            future.result()
        exc = cm.exception
        self.assertIs(type(exc), RuntimeError)
        self.assertEqual(exc.args, (123,))
        cause = exc.__cause__
        self.assertIs(type(cause), futures.process._RemoteTraceback)
        self.assertIn('raise RuntimeError(123) # some comment', cause.tb)
        with support.captured_stderr() as f1:
            try:
                raise exc
            except RuntimeError:
                sys.excepthook(*sys.exc_info())
        self.assertIn('raise RuntimeError(123) # some comment', f1.getvalue())

    @hashlib_helper.requires_hashdigest('md5')
    def test_ressources_gced_in_workers(self):
        if False:
            for i in range(10):
                print('nop')
        mgr = self.get_context().Manager()
        obj = EventfulGCObj(mgr)
        future = self.executor.submit(id, obj)
        future.result()
        self.assertTrue(obj.event.wait(timeout=1))
        obj = None
        support.gc_collect()
        mgr.shutdown()
        mgr.join()

    def test_saturation(self):
        if False:
            for i in range(10):
                print('nop')
        executor = self.executor
        mp_context = self.get_context()
        sem = mp_context.Semaphore(0)
        job_count = 15 * executor._max_workers
        for _ in range(job_count):
            executor.submit(sem.acquire)
        self.assertEqual(len(executor._processes), executor._max_workers)
        for _ in range(job_count):
            sem.release()

    def test_idle_process_reuse_one(self):
        if False:
            for i in range(10):
                print('nop')
        executor = self.executor
        assert executor._max_workers >= 4
        if self.get_context().get_start_method(allow_none=False) == 'fork':
            raise unittest.SkipTest('Incompatible with the fork start method.')
        executor.submit(mul, 21, 2).result()
        executor.submit(mul, 6, 7).result()
        executor.submit(mul, 3, 14).result()
        self.assertEqual(len(executor._processes), 1)

    def test_idle_process_reuse_multiple(self):
        if False:
            print('Hello World!')
        executor = self.executor
        assert executor._max_workers <= 5
        if self.get_context().get_start_method(allow_none=False) == 'fork':
            raise unittest.SkipTest('Incompatible with the fork start method.')
        executor.submit(mul, 12, 7).result()
        executor.submit(mul, 33, 25)
        executor.submit(mul, 25, 26).result()
        executor.submit(mul, 18, 29)
        executor.submit(mul, 1, 2).result()
        executor.submit(mul, 0, 9)
        self.assertLessEqual(len(executor._processes), 3)
        executor.shutdown()
create_executor_tests(ProcessPoolExecutorTest, executor_mixins=(ProcessPoolForkMixin, ProcessPoolForkserverMixin, ProcessPoolSpawnMixin))

def _crash(delay=None):
    if False:
        print('Hello World!')
    'Induces a segfault.'
    if delay:
        time.sleep(delay)
    import faulthandler
    faulthandler.disable()
    faulthandler._sigsegv()

def _exit():
    if False:
        while True:
            i = 10
    'Induces a sys exit with exitcode 1.'
    sys.exit(1)

def _raise_error(Err):
    if False:
        i = 10
        return i + 15
    'Function that raises an Exception in process.'
    raise Err()

def _raise_error_ignore_stderr(Err):
    if False:
        print('Hello World!')
    'Function that raises an Exception in process and ignores stderr.'
    import io
    sys.stderr = io.StringIO()
    raise Err()

def _return_instance(cls):
    if False:
        print('Hello World!')
    'Function that returns a instance of cls.'
    return cls()

def _identity(x):
    if False:
        print('Hello World!')
    return x

class CrashAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        _crash()

class CrashAtUnpickle(object):
    """Bad object that triggers a segfault at unpickling time."""

    def __reduce__(self):
        if False:
            return 10
        return (_crash, ())

class ExitAtPickle(object):
    """Bad object that triggers a process exit at pickling time."""

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        _exit()

class ExitAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""

    def __reduce__(self):
        if False:
            return 10
        return (_exit, ())

class ErrorAtPickle(object):
    """Bad object that triggers an error at pickling time."""

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        from pickle import PicklingError
        raise PicklingError('Error in pickle')

class ErrorAtUnpickle(object):
    """Bad object that triggers an error at unpickling time."""

    def __reduce__(self):
        if False:
            while True:
                i = 10
        from pickle import UnpicklingError
        return (_raise_error_ignore_stderr, (UnpicklingError,))

class ExecutorDeadlockTest:
    TIMEOUT = support.SHORT_TIMEOUT

    def _fail_on_deadlock(self, executor):
        if False:
            for i in range(10):
                print('nop')
        import faulthandler
        from tempfile import TemporaryFile
        with TemporaryFile(mode='w+') as f:
            faulthandler.dump_traceback(file=f)
            f.seek(0)
            tb = f.read()
        for p in executor._processes.values():
            p.terminate()
        executor.shutdown(wait=True)
        print(f'\nTraceback:\n {tb}', file=sys.__stderr__)
        self.fail(f'Executor deadlock:\n\n{tb}')

    def _check_crash(self, error, func, *args, ignore_stderr=False):
        if False:
            print('Hello World!')
        self.executor.shutdown(wait=True)
        executor = self.executor_type(max_workers=2, mp_context=self.get_context())
        res = executor.submit(func, *args)
        if ignore_stderr:
            cm = support.captured_stderr()
        else:
            cm = contextlib.nullcontext()
        try:
            with self.assertRaises(error):
                with cm:
                    res.result(timeout=self.TIMEOUT)
        except futures.TimeoutError:
            self._fail_on_deadlock(executor)
        executor.shutdown(wait=True)

    def test_error_at_task_pickle(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_crash(PicklingError, id, ErrorAtPickle())

    def test_exit_at_task_unpickle(self):
        if False:
            print('Hello World!')
        self._check_crash(BrokenProcessPool, id, ExitAtUnpickle())

    def test_error_at_task_unpickle(self):
        if False:
            print('Hello World!')
        self._check_crash(BrokenProcessPool, id, ErrorAtUnpickle())

    def test_crash_at_task_unpickle(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_crash(BrokenProcessPool, id, CrashAtUnpickle())

    def test_crash_during_func_exec_on_worker(self):
        if False:
            while True:
                i = 10
        self._check_crash(BrokenProcessPool, _crash)

    def test_exit_during_func_exec_on_worker(self):
        if False:
            i = 10
            return i + 15
        self._check_crash(SystemExit, _exit)

    def test_error_during_func_exec_on_worker(self):
        if False:
            print('Hello World!')
        self._check_crash(RuntimeError, _raise_error, RuntimeError)

    def test_crash_during_result_pickle_on_worker(self):
        if False:
            while True:
                i = 10
        self._check_crash(BrokenProcessPool, _return_instance, CrashAtPickle)

    def test_exit_during_result_pickle_on_worker(self):
        if False:
            while True:
                i = 10
        self._check_crash(SystemExit, _return_instance, ExitAtPickle)

    def test_error_during_result_pickle_on_worker(self):
        if False:
            i = 10
            return i + 15
        self._check_crash(PicklingError, _return_instance, ErrorAtPickle)

    def test_error_during_result_unpickle_in_result_handler(self):
        if False:
            while True:
                i = 10
        self._check_crash(BrokenProcessPool, _return_instance, ErrorAtUnpickle, ignore_stderr=True)

    def test_exit_during_result_unpickle_in_result_handler(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_crash(BrokenProcessPool, _return_instance, ExitAtUnpickle)

    def test_shutdown_deadlock(self):
        if False:
            while True:
                i = 10
        self.executor.shutdown(wait=True)
        with self.executor_type(max_workers=2, mp_context=self.get_context()) as executor:
            self.executor = executor
            f = executor.submit(_crash, delay=0.1)
            executor.shutdown(wait=True)
            with self.assertRaises(BrokenProcessPool):
                f.result()

    def test_shutdown_deadlock_pickle(self):
        if False:
            for i in range(10):
                print('nop')
        self.executor.shutdown(wait=True)
        with self.executor_type(max_workers=2, mp_context=self.get_context()) as executor:
            self.executor = executor
            executor.submit(id, 42).result()
            executor_manager = executor._executor_manager_thread
            f = executor.submit(id, ErrorAtPickle())
            executor.shutdown(wait=False)
            with self.assertRaises(PicklingError):
                f.result()
        executor_manager.join()

    def test_shutdown_deadlock_blocked_pipe(self):
        if False:
            for i in range(10):
                print('nop')
        self.executor.shutdown(wait=True)
        with self.executor_type(max_workers=1, mp_context=self.get_context()) as executor:
            f = executor.submit(_crash)
            f = executor.submit(_identity, 'a' * 8192000)
            with self.assertRaises(BrokenProcessPool):
                f.result()
            executor.shutdown(wait=True)
create_executor_tests(ExecutorDeadlockTest, executor_mixins=(ProcessPoolForkMixin, ProcessPoolForkserverMixin, ProcessPoolSpawnMixin))

class FutureTests(BaseTestCase):

    def test_done_callback_with_result(self):
        if False:
            for i in range(10):
                print('nop')
        callback_result = None

        def fn(callback_future):
            if False:
                while True:
                    i = 10
            nonlocal callback_result
            callback_result = callback_future.result()
        f = Future()
        f.add_done_callback(fn)
        f.set_result(5)
        self.assertEqual(5, callback_result)

    def test_done_callback_with_exception(self):
        if False:
            for i in range(10):
                print('nop')
        callback_exception = None

        def fn(callback_future):
            if False:
                return 10
            nonlocal callback_exception
            callback_exception = callback_future.exception()
        f = Future()
        f.add_done_callback(fn)
        f.set_exception(Exception('test'))
        self.assertEqual(('test',), callback_exception.args)

    def test_done_callback_with_cancel(self):
        if False:
            i = 10
            return i + 15
        was_cancelled = None

        def fn(callback_future):
            if False:
                i = 10
                return i + 15
            nonlocal was_cancelled
            was_cancelled = callback_future.cancelled()
        f = Future()
        f.add_done_callback(fn)
        self.assertTrue(f.cancel())
        self.assertTrue(was_cancelled)

    def test_done_callback_raises(self):
        if False:
            print('Hello World!')
        with support.captured_stderr() as stderr:
            raising_was_called = False
            fn_was_called = False

            def raising_fn(callback_future):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal raising_was_called
                raising_was_called = True
                raise Exception('doh!')

            def fn(callback_future):
                if False:
                    i = 10
                    return i + 15
                nonlocal fn_was_called
                fn_was_called = True
            f = Future()
            f.add_done_callback(raising_fn)
            f.add_done_callback(fn)
            f.set_result(5)
            self.assertTrue(raising_was_called)
            self.assertTrue(fn_was_called)
            self.assertIn('Exception: doh!', stderr.getvalue())

    def test_done_callback_already_successful(self):
        if False:
            for i in range(10):
                print('nop')
        callback_result = None

        def fn(callback_future):
            if False:
                i = 10
                return i + 15
            nonlocal callback_result
            callback_result = callback_future.result()
        f = Future()
        f.set_result(5)
        f.add_done_callback(fn)
        self.assertEqual(5, callback_result)

    def test_done_callback_already_failed(self):
        if False:
            print('Hello World!')
        callback_exception = None

        def fn(callback_future):
            if False:
                while True:
                    i = 10
            nonlocal callback_exception
            callback_exception = callback_future.exception()
        f = Future()
        f.set_exception(Exception('test'))
        f.add_done_callback(fn)
        self.assertEqual(('test',), callback_exception.args)

    def test_done_callback_already_cancelled(self):
        if False:
            while True:
                i = 10
        was_cancelled = None

        def fn(callback_future):
            if False:
                print('Hello World!')
            nonlocal was_cancelled
            was_cancelled = callback_future.cancelled()
        f = Future()
        self.assertTrue(f.cancel())
        f.add_done_callback(fn)
        self.assertTrue(was_cancelled)

    def test_done_callback_raises_already_succeeded(self):
        if False:
            for i in range(10):
                print('nop')
        with support.captured_stderr() as stderr:

            def raising_fn(callback_future):
                if False:
                    i = 10
                    return i + 15
                raise Exception('doh!')
            f = Future()
            f.set_result(5)
            f.add_done_callback(raising_fn)
            self.assertIn('exception calling callback for', stderr.getvalue())
            self.assertIn('doh!', stderr.getvalue())

    def test_repr(self):
        if False:
            print('Hello World!')
        self.assertRegex(repr(PENDING_FUTURE), '<Future at 0x[0-9a-f]+ state=pending>')
        self.assertRegex(repr(RUNNING_FUTURE), '<Future at 0x[0-9a-f]+ state=running>')
        self.assertRegex(repr(CANCELLED_FUTURE), '<Future at 0x[0-9a-f]+ state=cancelled>')
        self.assertRegex(repr(CANCELLED_AND_NOTIFIED_FUTURE), '<Future at 0x[0-9a-f]+ state=cancelled>')
        self.assertRegex(repr(EXCEPTION_FUTURE), '<Future at 0x[0-9a-f]+ state=finished raised OSError>')
        self.assertRegex(repr(SUCCESSFUL_FUTURE), '<Future at 0x[0-9a-f]+ state=finished returned int>')

    def test_cancel(self):
        if False:
            while True:
                i = 10
        f1 = create_future(state=PENDING)
        f2 = create_future(state=RUNNING)
        f3 = create_future(state=CANCELLED)
        f4 = create_future(state=CANCELLED_AND_NOTIFIED)
        f5 = create_future(state=FINISHED, exception=OSError())
        f6 = create_future(state=FINISHED, result=5)
        self.assertTrue(f1.cancel())
        self.assertEqual(f1._state, CANCELLED)
        self.assertFalse(f2.cancel())
        self.assertEqual(f2._state, RUNNING)
        self.assertTrue(f3.cancel())
        self.assertEqual(f3._state, CANCELLED)
        self.assertTrue(f4.cancel())
        self.assertEqual(f4._state, CANCELLED_AND_NOTIFIED)
        self.assertFalse(f5.cancel())
        self.assertEqual(f5._state, FINISHED)
        self.assertFalse(f6.cancel())
        self.assertEqual(f6._state, FINISHED)

    def test_cancelled(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(PENDING_FUTURE.cancelled())
        self.assertFalse(RUNNING_FUTURE.cancelled())
        self.assertTrue(CANCELLED_FUTURE.cancelled())
        self.assertTrue(CANCELLED_AND_NOTIFIED_FUTURE.cancelled())
        self.assertFalse(EXCEPTION_FUTURE.cancelled())
        self.assertFalse(SUCCESSFUL_FUTURE.cancelled())

    def test_done(self):
        if False:
            while True:
                i = 10
        self.assertFalse(PENDING_FUTURE.done())
        self.assertFalse(RUNNING_FUTURE.done())
        self.assertTrue(CANCELLED_FUTURE.done())
        self.assertTrue(CANCELLED_AND_NOTIFIED_FUTURE.done())
        self.assertTrue(EXCEPTION_FUTURE.done())
        self.assertTrue(SUCCESSFUL_FUTURE.done())

    def test_running(self):
        if False:
            while True:
                i = 10
        self.assertFalse(PENDING_FUTURE.running())
        self.assertTrue(RUNNING_FUTURE.running())
        self.assertFalse(CANCELLED_FUTURE.running())
        self.assertFalse(CANCELLED_AND_NOTIFIED_FUTURE.running())
        self.assertFalse(EXCEPTION_FUTURE.running())
        self.assertFalse(SUCCESSFUL_FUTURE.running())

    def test_result_with_timeout(self):
        if False:
            while True:
                i = 10
        self.assertRaises(futures.TimeoutError, PENDING_FUTURE.result, timeout=0)
        self.assertRaises(futures.TimeoutError, RUNNING_FUTURE.result, timeout=0)
        self.assertRaises(futures.CancelledError, CANCELLED_FUTURE.result, timeout=0)
        self.assertRaises(futures.CancelledError, CANCELLED_AND_NOTIFIED_FUTURE.result, timeout=0)
        self.assertRaises(OSError, EXCEPTION_FUTURE.result, timeout=0)
        self.assertEqual(SUCCESSFUL_FUTURE.result(timeout=0), 42)

    def test_result_with_success(self):
        if False:
            while True:
                i = 10

        def notification():
            if False:
                return 10
            time.sleep(1)
            f1.set_result(42)
        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()
        self.assertEqual(f1.result(timeout=5), 42)
        t.join()

    def test_result_with_cancel(self):
        if False:
            i = 10
            return i + 15

        def notification():
            if False:
                return 10
            time.sleep(1)
            f1.cancel()
        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()
        self.assertRaises(futures.CancelledError, f1.result, timeout=support.SHORT_TIMEOUT)
        t.join()

    def test_exception_with_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(futures.TimeoutError, PENDING_FUTURE.exception, timeout=0)
        self.assertRaises(futures.TimeoutError, RUNNING_FUTURE.exception, timeout=0)
        self.assertRaises(futures.CancelledError, CANCELLED_FUTURE.exception, timeout=0)
        self.assertRaises(futures.CancelledError, CANCELLED_AND_NOTIFIED_FUTURE.exception, timeout=0)
        self.assertTrue(isinstance(EXCEPTION_FUTURE.exception(timeout=0), OSError))
        self.assertEqual(SUCCESSFUL_FUTURE.exception(timeout=0), None)

    def test_exception_with_success(self):
        if False:
            while True:
                i = 10

        def notification():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(1)
            with f1._condition:
                f1._state = FINISHED
                f1._exception = OSError()
                f1._condition.notify_all()
        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()
        self.assertTrue(isinstance(f1.exception(timeout=support.SHORT_TIMEOUT), OSError))
        t.join()

    def test_multiple_set_result(self):
        if False:
            return 10
        f = create_future(state=PENDING)
        f.set_result(1)
        with self.assertRaisesRegex(futures.InvalidStateError, 'FINISHED: <Future at 0x[0-9a-f]+ state=finished returned int>'):
            f.set_result(2)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 1)

    def test_multiple_set_exception(self):
        if False:
            print('Hello World!')
        f = create_future(state=PENDING)
        e = ValueError()
        f.set_exception(e)
        with self.assertRaisesRegex(futures.InvalidStateError, 'FINISHED: <Future at 0x[0-9a-f]+ state=finished raised ValueError>'):
            f.set_exception(Exception())
        self.assertEqual(f.exception(), e)

def setUpModule():
    if False:
        for i in range(10):
            print('nop')
    unittest.addModuleCleanup(multiprocessing.util._cleanup_tests)
    thread_info = threading_helper.threading_setup()
    unittest.addModuleCleanup(threading_helper.threading_cleanup, *thread_info)
if __name__ == '__main__':
    unittest.main()