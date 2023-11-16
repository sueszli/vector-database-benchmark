import unittest
import unittest.mock
import queue as pyqueue
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
_multiprocessing = import_helper.import_module('_multiprocessing')
support.skip_if_broken_multiprocessing_synchronize()
import threading
import multiprocessing.connection
import multiprocessing.dummy
import multiprocessing.heap
import multiprocessing.managers
import multiprocessing.pool
import multiprocessing.queues
from multiprocessing import util
try:
    from multiprocessing import reduction
    HAS_REDUCTION = reduction.HAVE_SEND_HANDLE
except ImportError:
    HAS_REDUCTION = False
try:
    from multiprocessing.sharedctypes import Value, copy
    HAS_SHAREDCTYPES = True
except ImportError:
    HAS_SHAREDCTYPES = False
try:
    from multiprocessing import shared_memory
    HAS_SHMEM = True
except ImportError:
    HAS_SHMEM = False
try:
    import msvcrt
except ImportError:
    msvcrt = None
if support.check_sanitizer(address=True):
    raise unittest.SkipTest('libasan has a pthread_create() dead lock')

def latin(s):
    if False:
        print('Hello World!')
    return s.encode('latin')

def close_queue(queue):
    if False:
        i = 10
        return i + 15
    if isinstance(queue, multiprocessing.queues.Queue):
        queue.close()
        queue.join_thread()

def join_process(process):
    if False:
        while True:
            i = 10
    threading_helper.join_thread(process)
if os.name == 'posix':
    from multiprocessing import resource_tracker

    def _resource_unlink(name, rtype):
        if False:
            i = 10
            return i + 15
        resource_tracker._CLEANUP_FUNCS[rtype](name)
LOG_LEVEL = util.SUBWARNING
DELTA = 0.1
CHECK_TIMINGS = False
if CHECK_TIMINGS:
    (TIMEOUT1, TIMEOUT2, TIMEOUT3) = (0.82, 0.35, 1.4)
else:
    (TIMEOUT1, TIMEOUT2, TIMEOUT3) = (0.1, 0.1, 0.1)
HAVE_GETVALUE = not getattr(_multiprocessing, 'HAVE_BROKEN_SEM_GETVALUE', False)
WIN32 = sys.platform == 'win32'
from multiprocessing.connection import wait

def wait_for_handle(handle, timeout):
    if False:
        i = 10
        return i + 15
    if timeout is not None and timeout < 0.0:
        timeout = None
    return wait([handle], timeout)
try:
    MAXFD = os.sysconf('SC_OPEN_MAX')
except:
    MAXFD = 256
PRELOAD = ['__main__', 'test.test_multiprocessing_forkserver']
try:
    from ctypes import Structure, c_int, c_double, c_longlong
except ImportError:
    Structure = object
    c_int = c_double = c_longlong = None

def check_enough_semaphores():
    if False:
        print('Hello World!')
    'Check that the system supports enough semaphores to run the test.'
    nsems_min = 256
    try:
        nsems = os.sysconf('SC_SEM_NSEMS_MAX')
    except (AttributeError, ValueError):
        return
    if nsems == -1 or nsems >= nsems_min:
        return
    raise unittest.SkipTest("The OS doesn't support enough semaphores to run the test (required: %d)." % nsems_min)

class TimingWrapper(object):

    def __init__(self, func):
        if False:
            print('Hello World!')
        self.func = func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        t = time.monotonic()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = time.monotonic() - t

class BaseTestCase(object):
    ALLOWED_TYPES = ('processes', 'manager', 'threads')

    def assertTimingAlmostEqual(self, a, b):
        if False:
            print('Hello World!')
        if CHECK_TIMINGS:
            self.assertAlmostEqual(a, b, 1)

    def assertReturnsIfImplemented(self, value, func, *args):
        if False:
            return 10
        try:
            res = func(*args)
        except NotImplementedError:
            pass
        else:
            return self.assertEqual(value, res)

    def __reduce__(self, *args):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("shouldn't try to pickle a test case")
    __reduce_ex__ = __reduce__

def get_value(self):
    if False:
        while True:
            i = 10
    try:
        return self.get_value()
    except AttributeError:
        try:
            return self._Semaphore__value
        except AttributeError:
            try:
                return self._value
            except AttributeError:
                raise NotImplementedError

class DummyCallable:

    def __call__(self, q, c):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(c, DummyCallable)
        q.put(5)

class _TestProcess(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    def test_current(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        current = self.current_process()
        authkey = current.authkey
        self.assertTrue(current.is_alive())
        self.assertTrue(not current.daemon)
        self.assertIsInstance(authkey, bytes)
        self.assertTrue(len(authkey) > 0)
        self.assertEqual(current.ident, os.getpid())
        self.assertEqual(current.exitcode, None)

    def test_daemon_argument(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        proc0 = self.Process(target=self._test)
        self.assertEqual(proc0.daemon, self.current_process().daemon)
        proc1 = self.Process(target=self._test, daemon=True)
        self.assertTrue(proc1.daemon)
        proc2 = self.Process(target=self._test, daemon=False)
        self.assertFalse(proc2.daemon)

    @classmethod
    def _test(cls, q, *args, **kwds):
        if False:
            while True:
                i = 10
        current = cls.current_process()
        q.put(args)
        q.put(kwds)
        q.put(current.name)
        if cls.TYPE != 'threads':
            q.put(bytes(current.authkey))
            q.put(current.pid)

    def test_parent_process_attributes(self):
        if False:
            while True:
                i = 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        self.assertIsNone(self.parent_process())
        (rconn, wconn) = self.Pipe(duplex=False)
        p = self.Process(target=self._test_send_parent_process, args=(wconn,))
        p.start()
        p.join()
        (parent_pid, parent_name) = rconn.recv()
        self.assertEqual(parent_pid, self.current_process().pid)
        self.assertEqual(parent_pid, os.getpid())
        self.assertEqual(parent_name, self.current_process().name)

    @classmethod
    def _test_send_parent_process(cls, wconn):
        if False:
            i = 10
            return i + 15
        from multiprocessing.process import parent_process
        wconn.send([parent_process().pid, parent_process().name])

    def test_parent_process(self):
        if False:
            i = 10
            return i + 15
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        (rconn, wconn) = self.Pipe(duplex=False)
        p = self.Process(target=self._test_create_grandchild_process, args=(wconn,))
        p.start()
        if not rconn.poll(timeout=support.LONG_TIMEOUT):
            raise AssertionError('Could not communicate with child process')
        parent_process_status = rconn.recv()
        self.assertEqual(parent_process_status, 'alive')
        p.terminate()
        p.join()
        if not rconn.poll(timeout=support.LONG_TIMEOUT):
            raise AssertionError('Could not communicate with child process')
        parent_process_status = rconn.recv()
        self.assertEqual(parent_process_status, 'not alive')

    @classmethod
    def _test_create_grandchild_process(cls, wconn):
        if False:
            return 10
        p = cls.Process(target=cls._test_report_parent_status, args=(wconn,))
        p.start()
        time.sleep(300)

    @classmethod
    def _test_report_parent_status(cls, wconn):
        if False:
            while True:
                i = 10
        from multiprocessing.process import parent_process
        wconn.send('alive' if parent_process().is_alive() else 'not alive')
        parent_process().join(timeout=support.SHORT_TIMEOUT)
        wconn.send('alive' if parent_process().is_alive() else 'not alive')

    def test_process(self):
        if False:
            while True:
                i = 10
        q = self.Queue(1)
        e = self.Event()
        args = (q, 1, 2)
        kwargs = {'hello': 23, 'bye': 2.54}
        name = 'SomeProcess'
        p = self.Process(target=self._test, args=args, kwargs=kwargs, name=name)
        p.daemon = True
        current = self.current_process()
        if self.TYPE != 'threads':
            self.assertEqual(p.authkey, current.authkey)
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.daemon, True)
        self.assertNotIn(p, self.active_children())
        self.assertTrue(type(self.active_children()) is list)
        self.assertEqual(p.exitcode, None)
        p.start()
        self.assertEqual(p.exitcode, None)
        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())
        self.assertEqual(q.get(), args[1:])
        self.assertEqual(q.get(), kwargs)
        self.assertEqual(q.get(), p.name)
        if self.TYPE != 'threads':
            self.assertEqual(q.get(), current.authkey)
            self.assertEqual(q.get(), p.pid)
        p.join()
        self.assertEqual(p.exitcode, 0)
        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())
        close_queue(q)

    @unittest.skipUnless(threading._HAVE_THREAD_NATIVE_ID, 'needs native_id')
    def test_process_mainthread_native_id(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        current_mainthread_native_id = threading.main_thread().native_id
        q = self.Queue(1)
        p = self.Process(target=self._test_process_mainthread_native_id, args=(q,))
        p.start()
        child_mainthread_native_id = q.get()
        p.join()
        close_queue(q)
        self.assertNotEqual(current_mainthread_native_id, child_mainthread_native_id)

    @classmethod
    def _test_process_mainthread_native_id(cls, q):
        if False:
            print('Hello World!')
        mainthread_native_id = threading.main_thread().native_id
        q.put(mainthread_native_id)

    @classmethod
    def _sleep_some(cls):
        if False:
            while True:
                i = 10
        time.sleep(100)

    @classmethod
    def _test_sleep(cls, delay):
        if False:
            i = 10
            return i + 15
        time.sleep(delay)

    def _kill_process(self, meth):
        if False:
            i = 10
            return i + 15
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        p = self.Process(target=self._sleep_some)
        p.daemon = True
        p.start()
        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())
        self.assertEqual(p.exitcode, None)
        join = TimingWrapper(p.join)
        self.assertEqual(join(0), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)
        self.assertEqual(join(-1), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)
        time.sleep(1)
        meth(p)
        if hasattr(signal, 'alarm'):

            def handler(*args):
                if False:
                    for i in range(10):
                        print('nop')
                raise RuntimeError('join took too long: %s' % p)
            old_handler = signal.signal(signal.SIGALRM, handler)
            try:
                signal.alarm(10)
                self.assertEqual(join(), None)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            self.assertEqual(join(), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())
        p.join()
        return p.exitcode

    def test_terminate(self):
        if False:
            return 10
        exitcode = self._kill_process(multiprocessing.Process.terminate)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGTERM)

    def test_kill(self):
        if False:
            print('Hello World!')
        exitcode = self._kill_process(multiprocessing.Process.kill)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGKILL)

    def test_cpu_count(self):
        if False:
            i = 10
            return i + 15
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 1
        self.assertTrue(type(cpus) is int)
        self.assertTrue(cpus >= 1)

    def test_active_children(self):
        if False:
            return 10
        self.assertEqual(type(self.active_children()), list)
        p = self.Process(target=time.sleep, args=(DELTA,))
        self.assertNotIn(p, self.active_children())
        p.daemon = True
        p.start()
        self.assertIn(p, self.active_children())
        p.join()
        self.assertNotIn(p, self.active_children())

    @classmethod
    def _test_recursion(cls, wconn, id):
        if False:
            i = 10
            return i + 15
        wconn.send(id)
        if len(id) < 2:
            for i in range(2):
                p = cls.Process(target=cls._test_recursion, args=(wconn, id + [i]))
                p.start()
                p.join()

    def test_recursion(self):
        if False:
            print('Hello World!')
        (rconn, wconn) = self.Pipe(duplex=False)
        self._test_recursion(wconn, [])
        time.sleep(DELTA)
        result = []
        while rconn.poll():
            result.append(rconn.recv())
        expected = [[], [0], [0, 0], [0, 1], [1], [1, 0], [1, 1]]
        self.assertEqual(result, expected)

    @classmethod
    def _test_sentinel(cls, event):
        if False:
            print('Hello World!')
        event.wait(10.0)

    def test_sentinel(self):
        if False:
            while True:
                i = 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        event = self.Event()
        p = self.Process(target=self._test_sentinel, args=(event,))
        with self.assertRaises(ValueError):
            p.sentinel
        p.start()
        self.addCleanup(p.join)
        sentinel = p.sentinel
        self.assertIsInstance(sentinel, int)
        self.assertFalse(wait_for_handle(sentinel, timeout=0.0))
        event.set()
        p.join()
        self.assertTrue(wait_for_handle(sentinel, timeout=1))

    @classmethod
    def _test_close(cls, rc=0, q=None):
        if False:
            print('Hello World!')
        if q is not None:
            q.get()
        sys.exit(rc)

    def test_close(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        q = self.Queue()
        p = self.Process(target=self._test_close, kwargs={'q': q})
        p.daemon = True
        p.start()
        self.assertEqual(p.is_alive(), True)
        with self.assertRaises(ValueError):
            p.close()
        q.put(None)
        p.join()
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.exitcode, 0)
        p.close()
        with self.assertRaises(ValueError):
            p.is_alive()
        with self.assertRaises(ValueError):
            p.join()
        with self.assertRaises(ValueError):
            p.terminate()
        p.close()
        wr = weakref.ref(p)
        del p
        gc.collect()
        self.assertIs(wr(), None)
        close_queue(q)

    def test_many_processes(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        N = 5 if sm == 'spawn' else 100
        procs = [self.Process(target=self._test_sleep, args=(0.01,)) for i in range(N)]
        for p in procs:
            p.start()
        for p in procs:
            join_process(p)
        for p in procs:
            self.assertEqual(p.exitcode, 0)
        procs = [self.Process(target=self._sleep_some) for i in range(N)]
        for p in procs:
            p.start()
        time.sleep(0.001)
        for p in procs:
            p.terminate()
        for p in procs:
            join_process(p)
        if os.name != 'nt':
            exitcodes = [-signal.SIGTERM]
            if sys.platform == 'darwin':
                exitcodes.append(-signal.SIGKILL)
            for p in procs:
                self.assertIn(p.exitcode, exitcodes)

    def test_lose_target_ref(self):
        if False:
            for i in range(10):
                print('nop')
        c = DummyCallable()
        wr = weakref.ref(c)
        q = self.Queue()
        p = self.Process(target=c, args=(q, c))
        del c
        p.start()
        p.join()
        gc.collect()
        self.assertIs(wr(), None)
        self.assertEqual(q.get(), 5)
        close_queue(q)

    @classmethod
    def _test_child_fd_inflation(self, evt, q):
        if False:
            i = 10
            return i + 15
        q.put(os_helper.fd_count())
        evt.wait()

    def test_child_fd_inflation(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        if sm == 'fork':
            self.skipTest('test not appropriate for {}'.format(sm))
        N = 5
        evt = self.Event()
        q = self.Queue()
        procs = [self.Process(target=self._test_child_fd_inflation, args=(evt, q)) for i in range(N)]
        for p in procs:
            p.start()
        try:
            fd_counts = [q.get() for i in range(N)]
            self.assertEqual(len(set(fd_counts)), 1, fd_counts)
        finally:
            evt.set()
            for p in procs:
                p.join()
            close_queue(q)

    @classmethod
    def _test_wait_for_threads(self, evt):
        if False:
            print('Hello World!')

        def func1():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(0.5)
            evt.set()

        def func2():
            if False:
                i = 10
                return i + 15
            time.sleep(20)
            evt.clear()
        threading.Thread(target=func1).start()
        threading.Thread(target=func2, daemon=True).start()

    def test_wait_for_threads(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        evt = self.Event()
        proc = self.Process(target=self._test_wait_for_threads, args=(evt,))
        proc.start()
        proc.join()
        self.assertTrue(evt.is_set())

    @classmethod
    def _test_error_on_stdio_flush(self, evt, break_std_streams={}):
        if False:
            return 10
        for (stream_name, action) in break_std_streams.items():
            if action == 'close':
                stream = io.StringIO()
                stream.close()
            else:
                assert action == 'remove'
                stream = None
            setattr(sys, stream_name, None)
        evt.set()

    def test_error_on_stdio_flush_1(self):
        if False:
            print('Hello World!')
        streams = [io.StringIO(), None]
        streams[0].close()
        for stream_name in ('stdout', 'stderr'):
            for stream in streams:
                old_stream = getattr(sys, stream_name)
                setattr(sys, stream_name, stream)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush, args=(evt,))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    def test_error_on_stdio_flush_2(self):
        if False:
            while True:
                i = 10
        for stream_name in ('stdout', 'stderr'):
            for action in ('close', 'remove'):
                old_stream = getattr(sys, stream_name)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush, args=(evt, {stream_name: action}))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    @classmethod
    def _sleep_and_set_event(self, evt, delay=0.0):
        if False:
            while True:
                i = 10
        time.sleep(delay)
        evt.set()

    def check_forkserver_death(self, signum):
        if False:
            print('Hello World!')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        if sm != 'forkserver':
            self.skipTest('test not appropriate for {}'.format(sm))
        from multiprocessing.forkserver import _forkserver
        _forkserver.ensure_running()
        delay = 0.5
        evt = self.Event()
        proc = self.Process(target=self._sleep_and_set_event, args=(evt, delay))
        proc.start()
        pid = _forkserver._forkserver_pid
        os.kill(pid, signum)
        time.sleep(delay * 2.0)
        evt2 = self.Event()
        proc2 = self.Process(target=self._sleep_and_set_event, args=(evt2,))
        proc2.start()
        proc2.join()
        self.assertTrue(evt2.is_set())
        self.assertEqual(proc2.exitcode, 0)
        proc.join()
        self.assertTrue(evt.is_set())
        self.assertIn(proc.exitcode, (0, 255))

    def test_forkserver_sigint(self):
        if False:
            while True:
                i = 10
        self.check_forkserver_death(signal.SIGINT)

    def test_forkserver_sigkill(self):
        if False:
            for i in range(10):
                print('nop')
        if os.name != 'nt':
            self.check_forkserver_death(signal.SIGKILL)

class _UpperCaser(multiprocessing.Process):

    def __init__(self):
        if False:
            print('Hello World!')
        multiprocessing.Process.__init__(self)
        (self.child_conn, self.parent_conn) = multiprocessing.Pipe()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.parent_conn.close()
        for s in iter(self.child_conn.recv, None):
            self.child_conn.send(s.upper())
        self.child_conn.close()

    def submit(self, s):
        if False:
            for i in range(10):
                print('nop')
        assert type(s) is str
        self.parent_conn.send(s)
        return self.parent_conn.recv()

    def stop(self):
        if False:
            print('Hello World!')
        self.parent_conn.send(None)
        self.parent_conn.close()
        self.child_conn.close()

class _TestSubclassingProcess(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_subclassing(self):
        if False:
            for i in range(10):
                print('nop')
        uppercaser = _UpperCaser()
        uppercaser.daemon = True
        uppercaser.start()
        self.assertEqual(uppercaser.submit('hello'), 'HELLO')
        self.assertEqual(uppercaser.submit('world'), 'WORLD')
        uppercaser.stop()
        uppercaser.join()

    def test_stderr_flush(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        testfn = os_helper.TESTFN
        self.addCleanup(os_helper.unlink, testfn)
        proc = self.Process(target=self._test_stderr_flush, args=(testfn,))
        proc.start()
        proc.join()
        with open(testfn, encoding='utf-8') as f:
            err = f.read()
            self.assertIn('ZeroDivisionError', err)
            self.assertIn('test_multiprocessing.py', err)
            self.assertIn('1/0 # MARKER', err)

    @classmethod
    def _test_stderr_flush(cls, testfn):
        if False:
            return 10
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', encoding='utf-8', closefd=False)
        1 / 0

    @classmethod
    def _test_sys_exit(cls, reason, testfn):
        if False:
            for i in range(10):
                print('nop')
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', encoding='utf-8', closefd=False)
        sys.exit(reason)

    def test_sys_exit(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        testfn = os_helper.TESTFN
        self.addCleanup(os_helper.unlink, testfn)
        for reason in ([1, 2, 3], 'ignore this'):
            p = self.Process(target=self._test_sys_exit, args=(reason, testfn))
            p.daemon = True
            p.start()
            join_process(p)
            self.assertEqual(p.exitcode, 1)
            with open(testfn, encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content.rstrip(), str(reason))
            os.unlink(testfn)
        cases = [((True,), 1), ((False,), 0), ((8,), 8), ((None,), 0), ((), 0)]
        for (args, expected) in cases:
            with self.subTest(args=args):
                p = self.Process(target=sys.exit, args=args)
                p.daemon = True
                p.start()
                join_process(p)
                self.assertEqual(p.exitcode, expected)

def queue_empty(q):
    if False:
        while True:
            i = 10
    if hasattr(q, 'empty'):
        return q.empty()
    else:
        return q.qsize() == 0

def queue_full(q, maxsize):
    if False:
        i = 10
        return i + 15
    if hasattr(q, 'full'):
        return q.full()
    else:
        return q.qsize() == maxsize

class _TestQueue(BaseTestCase):

    @classmethod
    def _test_put(cls, queue, child_can_start, parent_can_continue):
        if False:
            i = 10
            return i + 15
        child_can_start.wait()
        for i in range(6):
            queue.get()
        parent_can_continue.set()

    def test_put(self):
        if False:
            return 10
        MAXSIZE = 6
        queue = self.Queue(maxsize=MAXSIZE)
        child_can_start = self.Event()
        parent_can_continue = self.Event()
        proc = self.Process(target=self._test_put, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)
        queue.put(1)
        queue.put(2, True)
        queue.put(3, True, None)
        queue.put(4, False)
        queue.put(5, False, None)
        queue.put_nowait(6)
        time.sleep(DELTA)
        self.assertEqual(queue_empty(queue), False)
        self.assertEqual(queue_full(queue, MAXSIZE), True)
        put = TimingWrapper(queue.put)
        put_nowait = TimingWrapper(queue.put_nowait)
        self.assertRaises(pyqueue.Full, put, 7, False)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, False, None)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put_nowait, 7)
        self.assertTimingAlmostEqual(put_nowait.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, True, TIMEOUT1)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT1)
        self.assertRaises(pyqueue.Full, put, 7, False, TIMEOUT2)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, True, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT3)
        child_can_start.set()
        parent_can_continue.wait()
        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)
        proc.join()
        close_queue(queue)

    @classmethod
    def _test_get(cls, queue, child_can_start, parent_can_continue):
        if False:
            return 10
        child_can_start.wait()
        queue.put(2)
        queue.put(3)
        queue.put(4)
        queue.put(5)
        parent_can_continue.set()

    def test_get(self):
        if False:
            i = 10
            return i + 15
        queue = self.Queue()
        child_can_start = self.Event()
        parent_can_continue = self.Event()
        proc = self.Process(target=self._test_get, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertEqual(queue_empty(queue), True)
        child_can_start.set()
        parent_can_continue.wait()
        time.sleep(DELTA)
        self.assertEqual(queue_empty(queue), False)
        self.assertEqual(queue.get(True, None), 2)
        self.assertEqual(queue.get(True), 3)
        self.assertEqual(queue.get(timeout=1), 4)
        self.assertEqual(queue.get_nowait(), 5)
        self.assertEqual(queue_empty(queue), True)
        get = TimingWrapper(queue.get)
        get_nowait = TimingWrapper(queue.get_nowait)
        self.assertRaises(pyqueue.Empty, get, False)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, False, None)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get_nowait)
        self.assertTimingAlmostEqual(get_nowait.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, True, TIMEOUT1)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)
        self.assertRaises(pyqueue.Empty, get, False, TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT3)
        proc.join()
        close_queue(queue)

    @classmethod
    def _test_fork(cls, queue):
        if False:
            while True:
                i = 10
        for i in range(10, 20):
            queue.put(i)

    def test_fork(self):
        if False:
            i = 10
            return i + 15
        queue = self.Queue()
        for i in range(10):
            queue.put(i)
        time.sleep(DELTA)
        p = self.Process(target=self._test_fork, args=(queue,))
        p.daemon = True
        p.start()
        for i in range(20):
            self.assertEqual(queue.get(), i)
        self.assertRaises(pyqueue.Empty, queue.get, False)
        p.join()
        close_queue(queue)

    def test_qsize(self):
        if False:
            while True:
                i = 10
        q = self.Queue()
        try:
            self.assertEqual(q.qsize(), 0)
        except NotImplementedError:
            self.skipTest('qsize method not implemented')
        q.put(1)
        self.assertEqual(q.qsize(), 1)
        q.put(5)
        self.assertEqual(q.qsize(), 2)
        q.get()
        self.assertEqual(q.qsize(), 1)
        q.get()
        self.assertEqual(q.qsize(), 0)
        close_queue(q)

    @classmethod
    def _test_task_done(cls, q):
        if False:
            return 10
        for obj in iter(q.get, None):
            time.sleep(DELTA)
            q.task_done()

    def test_task_done(self):
        if False:
            return 10
        queue = self.JoinableQueue()
        workers = [self.Process(target=self._test_task_done, args=(queue,)) for i in range(4)]
        for p in workers:
            p.daemon = True
            p.start()
        for i in range(10):
            queue.put(i)
        queue.join()
        for p in workers:
            queue.put(None)
        for p in workers:
            p.join()
        close_queue(queue)

    def test_no_import_lock_contention(self):
        if False:
            for i in range(10):
                print('nop')
        with os_helper.temp_cwd():
            module_name = 'imported_by_an_imported_module'
            with open(module_name + '.py', 'w', encoding='utf-8') as f:
                f.write("if 1:\n                    import multiprocessing\n\n                    q = multiprocessing.Queue()\n                    q.put('knock knock')\n                    q.get(timeout=3)\n                    q.close()\n                    del q\n                ")
            with import_helper.DirsOnSysPath(os.getcwd()):
                try:
                    __import__(module_name)
                except pyqueue.Empty:
                    self.fail('Probable regression on import lock contention; see Issue #22853')

    def test_timeout(self):
        if False:
            while True:
                i = 10
        q = multiprocessing.Queue()
        start = time.monotonic()
        self.assertRaises(pyqueue.Empty, q.get, True, 0.2)
        delta = time.monotonic() - start
        self.assertGreaterEqual(delta, 0.1)
        close_queue(q)

    def test_queue_feeder_donot_stop_onexc(self):
        if False:
            i = 10
            return i + 15
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):

            def __reduce__(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise AttributeError
        with test.support.captured_stderr():
            q = self.Queue()
            q.put(NotSerializable())
            q.put(True)
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
            close_queue(q)
        with test.support.captured_stderr():
            q = self.Queue(maxsize=1)
            q.put(NotSerializable())
            q.put(True)
            try:
                self.assertEqual(q.qsize(), 1)
            except NotImplementedError:
                pass
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
            self.assertTrue(q.empty())
            close_queue(q)

    def test_queue_feeder_on_queue_feeder_error(self):
        if False:
            print('Hello World!')
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):
            """Mock unserializable object"""

            def __init__(self):
                if False:
                    return 10
                self.reduce_was_called = False
                self.on_queue_feeder_error_was_called = False

            def __reduce__(self):
                if False:
                    return 10
                self.reduce_was_called = True
                raise AttributeError

        class SafeQueue(multiprocessing.queues.Queue):
            """Queue with overloaded _on_queue_feeder_error hook"""

            @staticmethod
            def _on_queue_feeder_error(e, obj):
                if False:
                    i = 10
                    return i + 15
                if isinstance(e, AttributeError) and isinstance(obj, NotSerializable):
                    obj.on_queue_feeder_error_was_called = True
        not_serializable_obj = NotSerializable()
        with test.support.captured_stderr():
            q = SafeQueue(ctx=multiprocessing.get_context())
            q.put(not_serializable_obj)
            q.put(True)
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
        self.assertTrue(not_serializable_obj.reduce_was_called)
        self.assertTrue(not_serializable_obj.on_queue_feeder_error_was_called)

    def test_closed_queue_put_get_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        for q in (multiprocessing.Queue(), multiprocessing.JoinableQueue()):
            q.close()
            with self.assertRaisesRegex(ValueError, 'is closed'):
                q.put('foo')
            with self.assertRaisesRegex(ValueError, 'is closed'):
                q.get()

class _TestLock(BaseTestCase):

    def test_lock(self):
        if False:
            print('Hello World!')
        lock = self.Lock()
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(False), False)
        self.assertEqual(lock.release(), None)
        self.assertRaises((ValueError, threading.ThreadError), lock.release)

    def test_rlock(self):
        if False:
            print('Hello World!')
        lock = self.RLock()
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.release(), None)
        self.assertEqual(lock.release(), None)
        self.assertEqual(lock.release(), None)
        self.assertRaises((AssertionError, RuntimeError), lock.release)

    def test_lock_context(self):
        if False:
            return 10
        with self.Lock():
            pass

class _TestSemaphore(BaseTestCase):

    def _test_semaphore(self, sem):
        if False:
            print('Hello World!')
        self.assertReturnsIfImplemented(2, get_value, sem)
        self.assertEqual(sem.acquire(), True)
        self.assertReturnsIfImplemented(1, get_value, sem)
        self.assertEqual(sem.acquire(), True)
        self.assertReturnsIfImplemented(0, get_value, sem)
        self.assertEqual(sem.acquire(False), False)
        self.assertReturnsIfImplemented(0, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(1, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(2, get_value, sem)

    def test_semaphore(self):
        if False:
            while True:
                i = 10
        sem = self.Semaphore(2)
        self._test_semaphore(sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(3, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(4, get_value, sem)

    def test_bounded_semaphore(self):
        if False:
            for i in range(10):
                print('nop')
        sem = self.BoundedSemaphore(2)
        self._test_semaphore(sem)

    def test_timeout(self):
        if False:
            while True:
                i = 10
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sem = self.Semaphore(0)
        acquire = TimingWrapper(sem.acquire)
        self.assertEqual(acquire(False), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0.0)
        self.assertEqual(acquire(False, None), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0.0)
        self.assertEqual(acquire(False, TIMEOUT1), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0)
        self.assertEqual(acquire(True, TIMEOUT2), False)
        self.assertTimingAlmostEqual(acquire.elapsed, TIMEOUT2)
        self.assertEqual(acquire(timeout=TIMEOUT3), False)
        self.assertTimingAlmostEqual(acquire.elapsed, TIMEOUT3)

class _TestCondition(BaseTestCase):

    @classmethod
    def f(cls, cond, sleeping, woken, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        cond.acquire()
        sleeping.release()
        cond.wait(timeout)
        woken.release()
        cond.release()

    def assertReachesEventually(self, func, value):
        if False:
            return 10
        for i in range(10):
            try:
                if func() == value:
                    break
            except NotImplementedError:
                break
            time.sleep(DELTA)
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(value, func)

    def check_invariant(self, cond):
        if False:
            print('Hello World!')
        if self.TYPE == 'processes':
            try:
                sleepers = cond._sleeping_count.get_value() - cond._woken_count.get_value()
                self.assertEqual(sleepers, 0)
                self.assertEqual(cond._wait_semaphore.get_value(), 0)
            except NotImplementedError:
                pass

    def test_notify(self):
        if False:
            while True:
                i = 10
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        p = self.Process(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)
        p = threading.Thread(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)
        sleeping.acquire()
        sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify()
        cond.release()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(1, get_value, woken)
        cond.acquire()
        cond.notify()
        cond.release()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(2, get_value, woken)
        self.check_invariant(cond)
        p.join()

    def test_notify_all(self):
        if False:
            i = 10
            return i + 15
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken, TIMEOUT1))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken, TIMEOUT1))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        for i in range(6):
            woken.acquire()
        self.assertReturnsIfImplemented(0, get_value, woken)
        self.check_invariant(cond)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify_all()
        cond.release()
        self.assertReachesEventually(lambda : get_value(woken), 6)
        self.check_invariant(cond)

    def test_notify_n(self):
        if False:
            while True:
                i = 10
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify(n=2)
        cond.release()
        self.assertReachesEventually(lambda : get_value(woken), 2)
        cond.acquire()
        cond.notify(n=4)
        cond.release()
        self.assertReachesEventually(lambda : get_value(woken), 6)
        cond.acquire()
        cond.notify(n=3)
        cond.release()
        self.assertReturnsIfImplemented(6, get_value, woken)
        self.check_invariant(cond)

    def test_timeout(self):
        if False:
            print('Hello World!')
        cond = self.Condition()
        wait = TimingWrapper(cond.wait)
        cond.acquire()
        res = wait(TIMEOUT1)
        cond.release()
        self.assertEqual(res, False)
        self.assertTimingAlmostEqual(wait.elapsed, TIMEOUT1)

    @classmethod
    def _test_waitfor_f(cls, cond, state):
        if False:
            i = 10
            return i + 15
        with cond:
            state.value = 0
            cond.notify()
            result = cond.wait_for(lambda : state.value == 4)
            if not result or state.value != 4:
                sys.exit(1)

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor(self):
        if False:
            while True:
                i = 10
        cond = self.Condition()
        state = self.Value('i', -1)
        p = self.Process(target=self._test_waitfor_f, args=(cond, state))
        p.daemon = True
        p.start()
        with cond:
            result = cond.wait_for(lambda : state.value == 0)
            self.assertTrue(result)
            self.assertEqual(state.value, 0)
        for i in range(4):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()
        join_process(p)
        self.assertEqual(p.exitcode, 0)

    @classmethod
    def _test_waitfor_timeout_f(cls, cond, state, success, sem):
        if False:
            i = 10
            return i + 15
        sem.release()
        with cond:
            expected = 0.1
            dt = time.monotonic()
            result = cond.wait_for(lambda : state.value == 4, timeout=expected)
            dt = time.monotonic() - dt
            if not result and expected * 0.6 < dt < expected * 10.0:
                success.value = True

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor_timeout(self):
        if False:
            while True:
                i = 10
        cond = self.Condition()
        state = self.Value('i', 0)
        success = self.Value('i', False)
        sem = self.Semaphore(0)
        p = self.Process(target=self._test_waitfor_timeout_f, args=(cond, state, success, sem))
        p.daemon = True
        p.start()
        self.assertTrue(sem.acquire(timeout=support.LONG_TIMEOUT))
        for i in range(3):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()
        join_process(p)
        self.assertTrue(success.value)

    @classmethod
    def _test_wait_result(cls, c, pid):
        if False:
            print('Hello World!')
        with c:
            c.notify()
        time.sleep(1)
        if pid is not None:
            os.kill(pid, signal.SIGINT)

    def test_wait_result(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self, ProcessesMixin) and sys.platform != 'win32':
            pid = os.getpid()
        else:
            pid = None
        c = self.Condition()
        with c:
            self.assertFalse(c.wait(0))
            self.assertFalse(c.wait(0.1))
            p = self.Process(target=self._test_wait_result, args=(c, pid))
            p.start()
            self.assertTrue(c.wait(60))
            if pid is not None:
                self.assertRaises(KeyboardInterrupt, c.wait, 60)
            p.join()

class _TestEvent(BaseTestCase):

    @classmethod
    def _test_event(cls, event):
        if False:
            print('Hello World!')
        time.sleep(TIMEOUT2)
        event.set()

    def test_event(self):
        if False:
            print('Hello World!')
        event = self.Event()
        wait = TimingWrapper(event.wait)
        self.assertEqual(event.is_set(), False)
        self.assertEqual(wait(0.0), False)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        self.assertEqual(wait(TIMEOUT1), False)
        self.assertTimingAlmostEqual(wait.elapsed, TIMEOUT1)
        event.set()
        self.assertEqual(event.is_set(), True)
        self.assertEqual(wait(), True)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        self.assertEqual(wait(TIMEOUT1), True)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        event.clear()
        p = self.Process(target=self._test_event, args=(event,))
        p.daemon = True
        p.start()
        self.assertEqual(wait(), True)
        p.join()

class _DummyList(object):

    def __init__(self):
        if False:
            return 10
        wrapper = multiprocessing.heap.BufferWrapper(struct.calcsize('i'))
        lock = multiprocessing.Lock()
        self.__setstate__((wrapper, lock))
        self._lengthbuf[0] = 0

    def __setstate__(self, state):
        if False:
            return 10
        (self._wrapper, self._lock) = state
        self._lengthbuf = self._wrapper.create_memoryview().cast('i')

    def __getstate__(self):
        if False:
            return 10
        return (self._wrapper, self._lock)

    def append(self, _):
        if False:
            return 10
        with self._lock:
            self._lengthbuf[0] += 1

    def __len__(self):
        if False:
            return 10
        with self._lock:
            return self._lengthbuf[0]

def _wait():
    if False:
        i = 10
        return i + 15
    time.sleep(0.01)

class Bunch(object):
    """
    A bunch of threads.
    """

    def __init__(self, namespace, f, args, n, wait_before_exit=False):
        if False:
            return 10
        "\n        Construct a bunch of `n` threads running the same function `f`.\n        If `wait_before_exit` is True, the threads won't terminate until\n        do_finish() is called.\n        "
        self.f = f
        self.args = args
        self.n = n
        self.started = namespace.DummyList()
        self.finished = namespace.DummyList()
        self._can_exit = namespace.Event()
        if not wait_before_exit:
            self._can_exit.set()
        threads = []
        for i in range(n):
            p = namespace.Process(target=self.task)
            p.daemon = True
            p.start()
            threads.append(p)

        def finalize(threads):
            if False:
                for i in range(10):
                    print('nop')
            for p in threads:
                p.join()
        self._finalizer = weakref.finalize(self, finalize, threads)

    def task(self):
        if False:
            i = 10
            return i + 15
        pid = os.getpid()
        self.started.append(pid)
        try:
            self.f(*self.args)
        finally:
            self.finished.append(pid)
            self._can_exit.wait(30)
            assert self._can_exit.is_set()

    def wait_for_started(self):
        if False:
            for i in range(10):
                print('nop')
        while len(self.started) < self.n:
            _wait()

    def wait_for_finished(self):
        if False:
            while True:
                i = 10
        while len(self.finished) < self.n:
            _wait()

    def do_finish(self):
        if False:
            print('Hello World!')
        self._can_exit.set()

    def close(self):
        if False:
            print('Hello World!')
        self._finalizer()

class AppendTrue(object):

    def __init__(self, obj):
        if False:
            while True:
                i = 10
        self.obj = obj

    def __call__(self):
        if False:
            return 10
        self.obj.append(True)

class _TestBarrier(BaseTestCase):
    """
    Tests for Barrier objects.
    """
    N = 5
    defaultTimeout = 30.0

    def setUp(self):
        if False:
            print('Hello World!')
        self.barrier = self.Barrier(self.N, timeout=self.defaultTimeout)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.barrier.abort()
        self.barrier = None

    def DummyList(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            return []
        elif self.TYPE == 'manager':
            return self.manager.list()
        else:
            return _DummyList()

    def run_threads(self, f, args):
        if False:
            while True:
                i = 10
        b = Bunch(self, f, args, self.N - 1)
        try:
            f(*args)
            b.wait_for_finished()
        finally:
            b.close()

    @classmethod
    def multipass(cls, barrier, results, n):
        if False:
            print('Hello World!')
        m = barrier.parties
        assert m == cls.N
        for i in range(n):
            results[0].append(True)
            assert len(results[1]) == i * m
            barrier.wait()
            results[1].append(True)
            assert len(results[0]) == (i + 1) * m
            barrier.wait()
        try:
            assert barrier.n_waiting == 0
        except NotImplementedError:
            pass
        assert not barrier.broken

    def test_barrier(self, passes=1):
        if False:
            i = 10
            return i + 15
        '\n        Test that a barrier is passed in lockstep\n        '
        results = [self.DummyList(), self.DummyList()]
        self.run_threads(self.multipass, (self.barrier, results, passes))

    def test_barrier_10(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a barrier works for 10 consecutive runs\n        '
        return self.test_barrier(10)

    @classmethod
    def _test_wait_return_f(cls, barrier, queue):
        if False:
            for i in range(10):
                print('nop')
        res = barrier.wait()
        queue.put(res)

    def test_wait_return(self):
        if False:
            while True:
                i = 10
        '\n        test the return value from barrier.wait\n        '
        queue = self.Queue()
        self.run_threads(self._test_wait_return_f, (self.barrier, queue))
        results = [queue.get() for i in range(self.N)]
        self.assertEqual(results.count(0), 1)
        close_queue(queue)

    @classmethod
    def _test_action_f(cls, barrier, results):
        if False:
            print('Hello World!')
        barrier.wait()
        if len(results) != 1:
            raise RuntimeError

    def test_action(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the 'action' callback\n        "
        results = self.DummyList()
        barrier = self.Barrier(self.N, action=AppendTrue(results))
        self.run_threads(self._test_action_f, (barrier, results))
        self.assertEqual(len(results), 1)

    @classmethod
    def _test_abort_f(cls, barrier, results1, results2):
        if False:
            while True:
                i = 10
        try:
            i = barrier.wait()
            if i == cls.N // 2:
                raise RuntimeError
            barrier.wait()
            results1.append(True)
        except threading.BrokenBarrierError:
            results2.append(True)
        except RuntimeError:
            barrier.abort()

    def test_abort(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that an abort will put the barrier in a broken state\n        '
        results1 = self.DummyList()
        results2 = self.DummyList()
        self.run_threads(self._test_abort_f, (self.barrier, results1, results2))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertTrue(self.barrier.broken)

    @classmethod
    def _test_reset_f(cls, barrier, results1, results2, results3):
        if False:
            for i in range(10):
                print('nop')
        i = barrier.wait()
        if i == cls.N // 2:
            while barrier.n_waiting < cls.N - 1:
                time.sleep(0.001)
            barrier.reset()
        else:
            try:
                barrier.wait()
                results1.append(True)
            except threading.BrokenBarrierError:
                results2.append(True)
        barrier.wait()
        results3.append(True)

    def test_reset(self):
        if False:
            while True:
                i = 10
        "\n        Test that a 'reset' on a barrier frees the waiting threads\n        "
        results1 = self.DummyList()
        results2 = self.DummyList()
        results3 = self.DummyList()
        self.run_threads(self._test_reset_f, (self.barrier, results1, results2, results3))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertEqual(len(results3), self.N)

    @classmethod
    def _test_abort_and_reset_f(cls, barrier, barrier2, results1, results2, results3):
        if False:
            for i in range(10):
                print('nop')
        try:
            i = barrier.wait()
            if i == cls.N // 2:
                raise RuntimeError
            barrier.wait()
            results1.append(True)
        except threading.BrokenBarrierError:
            results2.append(True)
        except RuntimeError:
            barrier.abort()
        if barrier2.wait() == cls.N // 2:
            barrier.reset()
        barrier2.wait()
        barrier.wait()
        results3.append(True)

    def test_abort_and_reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a barrier can be reset after being broken.\n        '
        results1 = self.DummyList()
        results2 = self.DummyList()
        results3 = self.DummyList()
        barrier2 = self.Barrier(self.N)
        self.run_threads(self._test_abort_and_reset_f, (self.barrier, barrier2, results1, results2, results3))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertEqual(len(results3), self.N)

    @classmethod
    def _test_timeout_f(cls, barrier, results):
        if False:
            while True:
                i = 10
        i = barrier.wait()
        if i == cls.N // 2:
            time.sleep(1.0)
        try:
            barrier.wait(0.5)
        except threading.BrokenBarrierError:
            results.append(True)

    def test_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test wait(timeout)\n        '
        results = self.DummyList()
        self.run_threads(self._test_timeout_f, (self.barrier, results))
        self.assertEqual(len(results), self.barrier.parties)

    @classmethod
    def _test_default_timeout_f(cls, barrier, results):
        if False:
            print('Hello World!')
        i = barrier.wait(cls.defaultTimeout)
        if i == cls.N // 2:
            time.sleep(1.0)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            results.append(True)

    def test_default_timeout(self):
        if False:
            print('Hello World!')
        "\n        Test the barrier's default timeout\n        "
        barrier = self.Barrier(self.N, timeout=0.5)
        results = self.DummyList()
        self.run_threads(self._test_default_timeout_f, (barrier, results))
        self.assertEqual(len(results), barrier.parties)

    def test_single_thread(self):
        if False:
            while True:
                i = 10
        b = self.Barrier(1)
        b.wait()
        b.wait()

    @classmethod
    def _test_thousand_f(cls, barrier, passes, conn, lock):
        if False:
            print('Hello World!')
        for i in range(passes):
            barrier.wait()
            with lock:
                conn.send(i)

    def test_thousand(self):
        if False:
            while True:
                i = 10
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        passes = 1000
        lock = self.Lock()
        (conn, child_conn) = self.Pipe(False)
        for j in range(self.N):
            p = self.Process(target=self._test_thousand_f, args=(self.barrier, passes, child_conn, lock))
            p.start()
            self.addCleanup(p.join)
        for i in range(passes):
            for j in range(self.N):
                self.assertEqual(conn.recv(), i)

class _TestValue(BaseTestCase):
    ALLOWED_TYPES = ('processes',)
    codes_values = [('i', 4343, 24234), ('d', 3.625, -4.25), ('h', -232, 234), ('q', 2 ** 33, 2 ** 34), ('c', latin('x'), latin('y'))]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if not HAS_SHAREDCTYPES:
            self.skipTest('requires multiprocessing.sharedctypes')

    @classmethod
    def _test(cls, values):
        if False:
            for i in range(10):
                print('nop')
        for (sv, cv) in zip(values, cls.codes_values):
            sv.value = cv[2]

    def test_value(self, raw=False):
        if False:
            for i in range(10):
                print('nop')
        if raw:
            values = [self.RawValue(code, value) for (code, value, _) in self.codes_values]
        else:
            values = [self.Value(code, value) for (code, value, _) in self.codes_values]
        for (sv, cv) in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[1])
        proc = self.Process(target=self._test, args=(values,))
        proc.daemon = True
        proc.start()
        proc.join()
        for (sv, cv) in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[2])

    def test_rawvalue(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_value(raw=True)

    def test_getobj_getlock(self):
        if False:
            i = 10
            return i + 15
        val1 = self.Value('i', 5)
        lock1 = val1.get_lock()
        obj1 = val1.get_obj()
        val2 = self.Value('i', 5, lock=None)
        lock2 = val2.get_lock()
        obj2 = val2.get_obj()
        lock = self.Lock()
        val3 = self.Value('i', 5, lock=lock)
        lock3 = val3.get_lock()
        obj3 = val3.get_obj()
        self.assertEqual(lock, lock3)
        arr4 = self.Value('i', 5, lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))
        self.assertRaises(AttributeError, self.Value, 'i', 5, lock='navalue')
        arr5 = self.RawValue('i', 5)
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))

class _TestArray(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def f(cls, seq):
        if False:
            print('Hello World!')
        for i in range(1, len(seq)):
            seq[i] += seq[i - 1]

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_array(self, raw=False):
        if False:
            while True:
                i = 10
        seq = [680, 626, 934, 821, 150, 233, 548, 982, 714, 831]
        if raw:
            arr = self.RawArray('i', seq)
        else:
            arr = self.Array('i', seq)
        self.assertEqual(len(arr), len(seq))
        self.assertEqual(arr[3], seq[3])
        self.assertEqual(list(arr[2:7]), list(seq[2:7]))
        arr[4:8] = seq[4:8] = array.array('i', [1, 2, 3, 4])
        self.assertEqual(list(arr[:]), seq)
        self.f(seq)
        p = self.Process(target=self.f, args=(arr,))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(list(arr[:]), seq)

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_array_from_size(self):
        if False:
            while True:
                i = 10
        size = 10
        for _ in range(3):
            arr = self.Array('i', size)
            self.assertEqual(len(arr), size)
            self.assertEqual(list(arr), [0] * size)
            arr[:] = range(10)
            self.assertEqual(list(arr), list(range(10)))
            del arr

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_rawarray(self):
        if False:
            print('Hello World!')
        self.test_array(raw=True)

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_getobj_getlock_obj(self):
        if False:
            while True:
                i = 10
        arr1 = self.Array('i', list(range(10)))
        lock1 = arr1.get_lock()
        obj1 = arr1.get_obj()
        arr2 = self.Array('i', list(range(10)), lock=None)
        lock2 = arr2.get_lock()
        obj2 = arr2.get_obj()
        lock = self.Lock()
        arr3 = self.Array('i', list(range(10)), lock=lock)
        lock3 = arr3.get_lock()
        obj3 = arr3.get_obj()
        self.assertEqual(lock, lock3)
        arr4 = self.Array('i', range(10), lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))
        self.assertRaises(AttributeError, self.Array, 'i', range(10), lock='notalock')
        arr5 = self.RawArray('i', range(10))
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))

class _TestContainers(BaseTestCase):
    ALLOWED_TYPES = ('manager',)

    def test_list(self):
        if False:
            while True:
                i = 10
        a = self.list(list(range(10)))
        self.assertEqual(a[:], list(range(10)))
        b = self.list()
        self.assertEqual(b[:], [])
        b.extend(list(range(5)))
        self.assertEqual(b[:], list(range(5)))
        self.assertEqual(b[2], 2)
        self.assertEqual(b[2:10], [2, 3, 4])
        b *= 2
        self.assertEqual(b[:], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(b + [5, 6], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(a[:], list(range(10)))
        d = [a, b]
        e = self.list(d)
        self.assertEqual([element[:] for element in e], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        f = self.list([a])
        a.append('hello')
        self.assertEqual(f[0][:], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'hello'])

    def test_list_iter(self):
        if False:
            print('Hello World!')
        a = self.list(list(range(10)))
        it = iter(a)
        self.assertEqual(list(it), list(range(10)))
        self.assertEqual(list(it), [])
        it = iter(a)
        a[0] = 100
        self.assertEqual(next(it), 100)

    def test_list_proxy_in_list(self):
        if False:
            for i in range(10):
                print('nop')
        a = self.list([self.list(range(3)) for _i in range(3)])
        self.assertEqual([inner[:] for inner in a], [[0, 1, 2]] * 3)
        a[0][-1] = 55
        self.assertEqual(a[0][:], [0, 1, 55])
        for i in range(1, 3):
            self.assertEqual(a[i][:], [0, 1, 2])
        self.assertEqual(a[1].pop(), 2)
        self.assertEqual(len(a[1]), 2)
        for i in range(0, 3, 2):
            self.assertEqual(len(a[i]), 3)
        del a
        b = self.list()
        b.append(b)
        del b

    def test_dict(self):
        if False:
            return 10
        d = self.dict()
        indices = list(range(65, 70))
        for i in indices:
            d[i] = chr(i)
        self.assertEqual(d.copy(), dict(((i, chr(i)) for i in indices)))
        self.assertEqual(sorted(d.keys()), indices)
        self.assertEqual(sorted(d.values()), [chr(i) for i in indices])
        self.assertEqual(sorted(d.items()), [(i, chr(i)) for i in indices])

    def test_dict_iter(self):
        if False:
            i = 10
            return i + 15
        d = self.dict()
        indices = list(range(65, 70))
        for i in indices:
            d[i] = chr(i)
        it = iter(d)
        self.assertEqual(list(it), indices)
        self.assertEqual(list(it), [])
        it = iter(d)
        d.clear()
        self.assertRaises(RuntimeError, next, it)

    def test_dict_proxy_nested(self):
        if False:
            i = 10
            return i + 15
        pets = self.dict(ferrets=2, hamsters=4)
        supplies = self.dict(water=10, feed=3)
        d = self.dict(pets=pets, supplies=supplies)
        self.assertEqual(supplies['water'], 10)
        self.assertEqual(d['supplies']['water'], 10)
        d['supplies']['blankets'] = 5
        self.assertEqual(supplies['blankets'], 5)
        self.assertEqual(d['supplies']['blankets'], 5)
        d['supplies']['water'] = 7
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(d['supplies']['water'], 7)
        del pets
        del supplies
        self.assertEqual(d['pets']['ferrets'], 2)
        d['supplies']['blankets'] = 11
        self.assertEqual(d['supplies']['blankets'], 11)
        pets = d['pets']
        supplies = d['supplies']
        supplies['water'] = 7
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(d['supplies']['water'], 7)
        d.clear()
        self.assertEqual(len(d), 0)
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(pets['hamsters'], 4)
        l = self.list([pets, supplies])
        l[0]['marmots'] = 1
        self.assertEqual(pets['marmots'], 1)
        self.assertEqual(l[0]['marmots'], 1)
        del pets
        del supplies
        self.assertEqual(l[0]['marmots'], 1)
        outer = self.list([[88, 99], l])
        self.assertIsInstance(outer[0], list)
        self.assertEqual(outer[-1][-1]['feed'], 3)

    def test_nested_queue(self):
        if False:
            while True:
                i = 10
        a = self.list()
        a.append(self.Queue())
        a[0].put(123)
        self.assertEqual(a[0].get(), 123)
        b = self.dict()
        b[0] = self.Queue()
        b[0].put(456)
        self.assertEqual(b[0].get(), 456)

    def test_namespace(self):
        if False:
            while True:
                i = 10
        n = self.Namespace()
        n.name = 'Bob'
        n.job = 'Builder'
        n._hidden = 'hidden'
        self.assertEqual((n.name, n.job), ('Bob', 'Builder'))
        del n.job
        self.assertEqual(str(n), "Namespace(name='Bob')")
        self.assertTrue(hasattr(n, 'name'))
        self.assertTrue(not hasattr(n, 'job'))

def sqr(x, wait=0.0):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(wait)
    return x * x

def mul(x, y):
    if False:
        i = 10
        return i + 15
    return x * y

def raise_large_valuerror(wait):
    if False:
        while True:
            i = 10
    time.sleep(wait)
    raise ValueError('x' * 1024 ** 2)

def identity(x):
    if False:
        for i in range(10):
            print('nop')
    return x

class CountedObject(object):
    n_instances = 0

    def __new__(cls):
        if False:
            i = 10
            return i + 15
        cls.n_instances += 1
        return object.__new__(cls)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        type(self).n_instances -= 1

class SayWhenError(ValueError):
    pass

def exception_throwing_generator(total, when):
    if False:
        print('Hello World!')
    if when == -1:
        raise SayWhenError('Somebody said when')
    for i in range(total):
        if i == when:
            raise SayWhenError('Somebody said when')
        yield i

class _TestPool(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.pool = cls.Pool(4)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.pool.terminate()
        cls.pool.join()
        cls.pool = None
        super().tearDownClass()

    def test_apply(self):
        if False:
            i = 10
            return i + 15
        papply = self.pool.apply
        self.assertEqual(papply(sqr, (5,)), sqr(5))
        self.assertEqual(papply(sqr, (), {'x': 3}), sqr(x=3))

    def test_map(self):
        if False:
            print('Hello World!')
        pmap = self.pool.map
        self.assertEqual(pmap(sqr, list(range(10))), list(map(sqr, list(range(10)))))
        self.assertEqual(pmap(sqr, list(range(100)), chunksize=20), list(map(sqr, list(range(100)))))

    def test_starmap(self):
        if False:
            i = 10
            return i + 15
        psmap = self.pool.starmap
        tuples = list(zip(range(10), range(9, -1, -1)))
        self.assertEqual(psmap(mul, tuples), list(itertools.starmap(mul, tuples)))
        tuples = list(zip(range(100), range(99, -1, -1)))
        self.assertEqual(psmap(mul, tuples, chunksize=20), list(itertools.starmap(mul, tuples)))

    def test_starmap_async(self):
        if False:
            print('Hello World!')
        tuples = list(zip(range(100), range(99, -1, -1)))
        self.assertEqual(self.pool.starmap_async(mul, tuples).get(), list(itertools.starmap(mul, tuples)))

    def test_map_async(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.pool.map_async(sqr, list(range(10))).get(), list(map(sqr, list(range(10)))))

    def test_map_async_callbacks(self):
        if False:
            print('Hello World!')
        call_args = self.manager.list() if self.TYPE == 'manager' else []
        self.pool.map_async(int, ['1'], callback=call_args.append, error_callback=call_args.append).wait()
        self.assertEqual(1, len(call_args))
        self.assertEqual([1], call_args[0])
        self.pool.map_async(int, ['a'], callback=call_args.append, error_callback=call_args.append).wait()
        self.assertEqual(2, len(call_args))
        self.assertIsInstance(call_args[1], ValueError)

    def test_map_unplicklable(self):
        if False:
            return 10
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class A(object):

            def __reduce__(self):
                if False:
                    print('Hello World!')
                raise RuntimeError('cannot pickle')
        with self.assertRaises(RuntimeError):
            self.pool.map(sqr, [A()] * 10)

    def test_map_chunksize(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.pool.map_async(sqr, [], chunksize=1).get(timeout=TIMEOUT1)
        except multiprocessing.TimeoutError:
            self.fail('pool.map_async with chunksize stalled on null list')

    def test_map_handle_iterable_exception(self):
        if False:
            print('Hello World!')
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(10, 3), 1)

        class SpecialIterable:

            def __iter__(self):
                if False:
                    print('Hello World!')
                return self

            def __next__(self):
                if False:
                    return 10
                raise SayWhenError

            def __len__(self):
                if False:
                    while True:
                        i = 10
                return 1
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)

    def test_async(self):
        if False:
            while True:
                i = 10
        res = self.pool.apply_async(sqr, (7, TIMEOUT1))
        get = TimingWrapper(res.get)
        self.assertEqual(get(), 49)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)

    def test_async_timeout(self):
        if False:
            i = 10
            return i + 15
        res = self.pool.apply_async(sqr, (6, TIMEOUT2 + 1.0))
        get = TimingWrapper(res.get)
        self.assertRaises(multiprocessing.TimeoutError, get, timeout=TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT2)

    def test_imap(self):
        if False:
            print('Hello World!')
        it = self.pool.imap(sqr, list(range(10)))
        self.assertEqual(list(it), list(map(sqr, list(range(10)))))
        it = self.pool.imap(sqr, list(range(10)))
        for i in range(10):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, it.__next__)
        it = self.pool.imap(sqr, list(range(1000)), chunksize=100)
        for i in range(1000):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, it.__next__)

    def test_imap_handle_iterable_exception(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(10, 3), 1)
        for i in range(3):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 2)
        for i in range(6):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 4)
        for i in range(4):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)

    def test_imap_unordered(self):
        if False:
            while True:
                i = 10
        it = self.pool.imap_unordered(sqr, list(range(10)))
        self.assertEqual(sorted(it), list(map(sqr, list(range(10)))))
        it = self.pool.imap_unordered(sqr, list(range(1000)), chunksize=100)
        self.assertEqual(sorted(it), list(map(sqr, list(range(1000)))))

    def test_imap_unordered_handle_iterable_exception(self):
        if False:
            return 10
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(10, 3), 1)
        expected_values = list(map(sqr, list(range(10))))
        with self.assertRaises(SayWhenError):
            for i in range(10):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(20, 7), 2)
        expected_values = list(map(sqr, list(range(20))))
        with self.assertRaises(SayWhenError):
            for i in range(20):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)

    def test_make_pool(self):
        if False:
            print('Hello World!')
        expected_error = RemoteError if self.TYPE == 'manager' else ValueError
        self.assertRaises(expected_error, self.Pool, -1)
        self.assertRaises(expected_error, self.Pool, 0)
        if self.TYPE != 'manager':
            p = self.Pool(3)
            try:
                self.assertEqual(3, len(p._pool))
            finally:
                p.close()
                p.join()

    def test_terminate(self):
        if False:
            while True:
                i = 10
        result = self.pool.map_async(time.sleep, [0.1 for i in range(10000)], chunksize=1)
        self.pool.terminate()
        join = TimingWrapper(self.pool.join)
        join()
        self.assertLess(join.elapsed, 2.0)

    def test_empty_iterable(self):
        if False:
            return 10
        p = self.Pool(1)
        self.assertEqual(p.map(sqr, []), [])
        self.assertEqual(list(p.imap(sqr, [])), [])
        self.assertEqual(list(p.imap_unordered(sqr, [])), [])
        self.assertEqual(p.map_async(sqr, []).get(), [])
        p.close()
        p.join()

    def test_context(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'processes':
            L = list(range(10))
            expected = [sqr(i) for i in L]
            with self.Pool(2) as p:
                r = p.map_async(sqr, L)
                self.assertEqual(r.get(), expected)
            p.join()
            self.assertRaises(ValueError, p.map_async, sqr, L)

    @classmethod
    def _test_traceback(cls):
        if False:
            return 10
        raise RuntimeError(123)

    def test_traceback(self):
        if False:
            i = 10
            return i + 15
        if self.TYPE == 'processes':
            with self.Pool(1) as p:
                try:
                    p.apply(self._test_traceback)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected RuntimeError')
            p.join()
            self.assertIs(type(exc), RuntimeError)
            self.assertEqual(exc.args, (123,))
            cause = exc.__cause__
            self.assertIs(type(cause), multiprocessing.pool.RemoteTraceback)
            self.assertIn('raise RuntimeError(123) # some comment', cause.tb)
            with test.support.captured_stderr() as f1:
                try:
                    raise exc
                except RuntimeError:
                    sys.excepthook(*sys.exc_info())
            self.assertIn('raise RuntimeError(123) # some comment', f1.getvalue())
            with self.Pool(1) as p:
                try:
                    p.map(sqr, exception_throwing_generator(1, -1), 1)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected SayWhenError')
                self.assertIs(type(exc), SayWhenError)
                self.assertIs(exc.__cause__, None)
            p.join()

    @classmethod
    def _test_wrapped_exception(cls):
        if False:
            return 10
        raise RuntimeError('foo')

    def test_wrapped_exception(self):
        if False:
            for i in range(10):
                print('nop')
        with self.Pool(1) as p:
            with self.assertRaises(RuntimeError):
                p.apply(self._test_wrapped_exception)
        p.join()

    def test_map_no_failfast(self):
        if False:
            print('Hello World!')
        t_start = time.monotonic()
        with self.assertRaises(ValueError):
            with self.Pool(2) as p:
                try:
                    p.map(raise_large_valuerror, [0, 1])
                finally:
                    time.sleep(0.5)
                    p.close()
                    p.join()
        self.assertGreater(time.monotonic() - t_start, 0.9)

    def test_release_task_refs(self):
        if False:
            print('Hello World!')
        objs = [CountedObject() for i in range(10)]
        refs = [weakref.ref(o) for o in objs]
        self.pool.map(identity, objs)
        del objs
        gc.collect()
        time.sleep(DELTA)
        self.assertEqual(set((wr() for wr in refs)), {None})
        self.assertEqual(CountedObject.n_instances, 0)

    def test_enter(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE == 'manager':
            self.skipTest('test not applicable to manager')
        pool = self.Pool(1)
        with pool:
            pass
        with self.assertRaises(ValueError):
            with pool:
                pass
        pool.join()

    def test_resource_warning(self):
        if False:
            print('Hello World!')
        if self.TYPE == 'manager':
            self.skipTest('test not applicable to manager')
        pool = self.Pool(1)
        pool.terminate()
        pool.join()
        pool._state = multiprocessing.pool.RUN
        with warnings_helper.check_warnings(('unclosed running multiprocessing pool', ResourceWarning)):
            pool = None
            support.gc_collect()

def raising():
    if False:
        return 10
    raise KeyError('key')

def unpickleable_result():
    if False:
        print('Hello World!')
    return lambda : 42

class _TestPoolWorkerErrors(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_async_error_callback(self):
        if False:
            print('Hello World!')
        p = multiprocessing.Pool(2)
        scratchpad = [None]

        def errback(exc):
            if False:
                for i in range(10):
                    print('nop')
            scratchpad[0] = exc
        res = p.apply_async(raising, error_callback=errback)
        self.assertRaises(KeyError, res.get)
        self.assertTrue(scratchpad[0])
        self.assertIsInstance(scratchpad[0], KeyError)
        p.close()
        p.join()

    def test_unpickleable_result(self):
        if False:
            return 10
        from multiprocessing.pool import MaybeEncodingError
        p = multiprocessing.Pool(2)
        for iteration in range(20):
            scratchpad = [None]

            def errback(exc):
                if False:
                    return 10
                scratchpad[0] = exc
            res = p.apply_async(unpickleable_result, error_callback=errback)
            self.assertRaises(MaybeEncodingError, res.get)
            wrapped = scratchpad[0]
            self.assertTrue(wrapped)
            self.assertIsInstance(scratchpad[0], MaybeEncodingError)
            self.assertIsNotNone(wrapped.exc)
            self.assertIsNotNone(wrapped.value)
        p.close()
        p.join()

class _TestPoolWorkerLifetime(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_pool_worker_lifetime(self):
        if False:
            i = 10
            return i + 15
        p = multiprocessing.Pool(3, maxtasksperchild=10)
        self.assertEqual(3, len(p._pool))
        origworkerpids = [w.pid for w in p._pool]
        results = []
        for i in range(100):
            results.append(p.apply_async(sqr, (i,)))
        for (j, res) in enumerate(results):
            self.assertEqual(res.get(), sqr(j))
        p._repopulate_pool()
        countdown = 50
        while countdown and (not all((w.is_alive() for w in p._pool))):
            countdown -= 1
            time.sleep(DELTA)
        finalworkerpids = [w.pid for w in p._pool]
        self.assertNotIn(None, origworkerpids)
        self.assertNotIn(None, finalworkerpids)
        self.assertNotEqual(sorted(origworkerpids), sorted(finalworkerpids))
        p.close()
        p.join()

    def test_pool_worker_lifetime_early_close(self):
        if False:
            for i in range(10):
                print('nop')
        p = multiprocessing.Pool(3, maxtasksperchild=1)
        results = []
        for i in range(6):
            results.append(p.apply_async(sqr, (i, 0.3)))
        p.close()
        p.join()
        for (j, res) in enumerate(results):
            self.assertEqual(res.get(), sqr(j))

    def test_worker_finalization_via_atexit_handler_of_multiprocessing(self):
        if False:
            return 10
        cmd = 'if 1:\n            from multiprocessing import Pool\n            problem = None\n            class A:\n                def __init__(self):\n                    self.pool = Pool(processes=1)\n            def test():\n                global problem\n                problem = A()\n                problem.pool.map(float, tuple(range(10)))\n            if __name__ == "__main__":\n                test()\n        '
        (rc, out, err) = test.support.script_helper.assert_python_ok('-c', cmd)
        self.assertEqual(rc, 0)
from multiprocessing.managers import BaseManager, BaseProxy, RemoteError

class FooBar(object):

    def f(self):
        if False:
            while True:
                i = 10
        return 'f()'

    def g(self):
        if False:
            while True:
                i = 10
        raise ValueError

    def _h(self):
        if False:
            return 10
        return '_h()'

def baz():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        yield (i * i)

class IteratorProxy(BaseProxy):
    _exposed_ = ('__next__',)

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._callmethod('__next__')

class MyManager(BaseManager):
    pass
MyManager.register('Foo', callable=FooBar)
MyManager.register('Bar', callable=FooBar, exposed=('f', '_h'))
MyManager.register('baz', callable=baz, proxytype=IteratorProxy)

class _TestMyManager(BaseTestCase):
    ALLOWED_TYPES = ('manager',)

    def test_mymanager(self):
        if False:
            while True:
                i = 10
        manager = MyManager()
        manager.start()
        self.common(manager)
        manager.shutdown()
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context(self):
        if False:
            for i in range(10):
                print('nop')
        with MyManager() as manager:
            self.common(manager)
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context_prestarted(self):
        if False:
            for i in range(10):
                print('nop')
        manager = MyManager()
        manager.start()
        with manager:
            self.common(manager)
        self.assertEqual(manager._process.exitcode, 0)

    def common(self, manager):
        if False:
            print('Hello World!')
        foo = manager.Foo()
        bar = manager.Bar()
        baz = manager.baz()
        foo_methods = [name for name in ('f', 'g', '_h') if hasattr(foo, name)]
        bar_methods = [name for name in ('f', 'g', '_h') if hasattr(bar, name)]
        self.assertEqual(foo_methods, ['f', 'g'])
        self.assertEqual(bar_methods, ['f', '_h'])
        self.assertEqual(foo.f(), 'f()')
        self.assertRaises(ValueError, foo.g)
        self.assertEqual(foo._callmethod('f'), 'f()')
        self.assertRaises(RemoteError, foo._callmethod, '_h')
        self.assertEqual(bar.f(), 'f()')
        self.assertEqual(bar._h(), '_h()')
        self.assertEqual(bar._callmethod('f'), 'f()')
        self.assertEqual(bar._callmethod('_h'), '_h()')
        self.assertEqual(list(baz), [i * i for i in range(10)])
_queue = pyqueue.Queue()

def get_queue():
    if False:
        return 10
    return _queue

class QueueManager(BaseManager):
    """manager class used by server process"""
QueueManager.register('get_queue', callable=get_queue)

class QueueManager2(BaseManager):
    """manager class which specifies the same interface as QueueManager"""
QueueManager2.register('get_queue')
SERIALIZER = 'xmlrpclib'

class _TestRemoteManager(BaseTestCase):
    ALLOWED_TYPES = ('manager',)
    values = ['hello world', None, True, 2.25, 'hallå världen', 'привіт світ', b'hall\xe5 v\xe4rlden']
    result = values[:]

    @classmethod
    def _putter(cls, address, authkey):
        if False:
            i = 10
            return i + 15
        manager = QueueManager2(address=address, authkey=authkey, serializer=SERIALIZER)
        manager.connect()
        queue = manager.get_queue()
        queue.put(tuple(cls.values))

    def test_remote(self):
        if False:
            for i in range(10):
                print('nop')
        authkey = os.urandom(32)
        manager = QueueManager(address=(socket_helper.HOST, 0), authkey=authkey, serializer=SERIALIZER)
        manager.start()
        self.addCleanup(manager.shutdown)
        p = self.Process(target=self._putter, args=(manager.address, authkey))
        p.daemon = True
        p.start()
        manager2 = QueueManager2(address=manager.address, authkey=authkey, serializer=SERIALIZER)
        manager2.connect()
        queue = manager2.get_queue()
        self.assertEqual(queue.get(), self.result)
        self.assertRaises(Exception, queue.put, time.sleep)
        del queue

@hashlib_helper.requires_hashdigest('md5')
class _TestManagerRestart(BaseTestCase):

    @classmethod
    def _putter(cls, address, authkey):
        if False:
            for i in range(10):
                print('nop')
        manager = QueueManager(address=address, authkey=authkey, serializer=SERIALIZER)
        manager.connect()
        queue = manager.get_queue()
        queue.put('hello world')

    def test_rapid_restart(self):
        if False:
            for i in range(10):
                print('nop')
        authkey = os.urandom(32)
        manager = QueueManager(address=(socket_helper.HOST, 0), authkey=authkey, serializer=SERIALIZER)
        try:
            srvr = manager.get_server()
            addr = srvr.address
            srvr.listener.close()
            manager.start()
            p = self.Process(target=self._putter, args=(manager.address, authkey))
            p.start()
            p.join()
            queue = manager.get_queue()
            self.assertEqual(queue.get(), 'hello world')
            del queue
        finally:
            if hasattr(manager, 'shutdown'):
                manager.shutdown()
        manager = QueueManager(address=addr, authkey=authkey, serializer=SERIALIZER)
        try:
            manager.start()
            self.addCleanup(manager.shutdown)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            time.sleep(1.0)
            manager = QueueManager(address=addr, authkey=authkey, serializer=SERIALIZER)
            if hasattr(manager, 'shutdown'):
                self.addCleanup(manager.shutdown)
SENTINEL = latin('')

class _TestConnection(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    @classmethod
    def _echo(cls, conn):
        if False:
            return 10
        for msg in iter(conn.recv_bytes, SENTINEL):
            conn.send_bytes(msg)
        conn.close()

    def test_connection(self):
        if False:
            while True:
                i = 10
        (conn, child_conn) = self.Pipe()
        p = self.Process(target=self._echo, args=(child_conn,))
        p.daemon = True
        p.start()
        seq = [1, 2.25, None]
        msg = latin('hello world')
        longmsg = msg * 10
        arr = array.array('i', list(range(4)))
        if self.TYPE == 'processes':
            self.assertEqual(type(conn.fileno()), int)
        self.assertEqual(conn.send(seq), None)
        self.assertEqual(conn.recv(), seq)
        self.assertEqual(conn.send_bytes(msg), None)
        self.assertEqual(conn.recv_bytes(), msg)
        if self.TYPE == 'processes':
            buffer = array.array('i', [0] * 10)
            expected = list(arr) + [0] * (10 - len(arr))
            self.assertEqual(conn.send_bytes(arr), None)
            self.assertEqual(conn.recv_bytes_into(buffer), len(arr) * buffer.itemsize)
            self.assertEqual(list(buffer), expected)
            buffer = array.array('i', [0] * 10)
            expected = [0] * 3 + list(arr) + [0] * (10 - 3 - len(arr))
            self.assertEqual(conn.send_bytes(arr), None)
            self.assertEqual(conn.recv_bytes_into(buffer, 3 * buffer.itemsize), len(arr) * buffer.itemsize)
            self.assertEqual(list(buffer), expected)
            buffer = bytearray(latin(' ' * 40))
            self.assertEqual(conn.send_bytes(longmsg), None)
            try:
                res = conn.recv_bytes_into(buffer)
            except multiprocessing.BufferTooShort as e:
                self.assertEqual(e.args, (longmsg,))
            else:
                self.fail('expected BufferTooShort, got %s' % res)
        poll = TimingWrapper(conn.poll)
        self.assertEqual(poll(), False)
        self.assertTimingAlmostEqual(poll.elapsed, 0)
        self.assertEqual(poll(-1), False)
        self.assertTimingAlmostEqual(poll.elapsed, 0)
        self.assertEqual(poll(TIMEOUT1), False)
        self.assertTimingAlmostEqual(poll.elapsed, TIMEOUT1)
        conn.send(None)
        time.sleep(0.1)
        self.assertEqual(poll(TIMEOUT1), True)
        self.assertTimingAlmostEqual(poll.elapsed, 0)
        self.assertEqual(conn.recv(), None)
        really_big_msg = latin('X') * (1024 * 1024 * 16)
        conn.send_bytes(really_big_msg)
        self.assertEqual(conn.recv_bytes(), really_big_msg)
        conn.send_bytes(SENTINEL)
        child_conn.close()
        if self.TYPE == 'processes':
            self.assertEqual(conn.readable, True)
            self.assertEqual(conn.writable, True)
            self.assertRaises(EOFError, conn.recv)
            self.assertRaises(EOFError, conn.recv_bytes)
        p.join()

    def test_duplex_false(self):
        if False:
            i = 10
            return i + 15
        (reader, writer) = self.Pipe(duplex=False)
        self.assertEqual(writer.send(1), None)
        self.assertEqual(reader.recv(), 1)
        if self.TYPE == 'processes':
            self.assertEqual(reader.readable, True)
            self.assertEqual(reader.writable, False)
            self.assertEqual(writer.readable, False)
            self.assertEqual(writer.writable, True)
            self.assertRaises(OSError, reader.send, 2)
            self.assertRaises(OSError, writer.recv)
            self.assertRaises(OSError, writer.poll)

    def test_spawn_close(self):
        if False:
            i = 10
            return i + 15
        (conn, child_conn) = self.Pipe()
        p = self.Process(target=self._echo, args=(child_conn,))
        p.daemon = True
        p.start()
        child_conn.close()
        msg = latin('hello')
        conn.send_bytes(msg)
        self.assertEqual(conn.recv_bytes(), msg)
        conn.send_bytes(SENTINEL)
        conn.close()
        p.join()

    def test_sendbytes(self):
        if False:
            return 10
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        msg = latin('abcdefghijklmnopqrstuvwxyz')
        (a, b) = self.Pipe()
        a.send_bytes(msg)
        self.assertEqual(b.recv_bytes(), msg)
        a.send_bytes(msg, 5)
        self.assertEqual(b.recv_bytes(), msg[5:])
        a.send_bytes(msg, 7, 8)
        self.assertEqual(b.recv_bytes(), msg[7:7 + 8])
        a.send_bytes(msg, 26)
        self.assertEqual(b.recv_bytes(), latin(''))
        a.send_bytes(msg, 26, 0)
        self.assertEqual(b.recv_bytes(), latin(''))
        self.assertRaises(ValueError, a.send_bytes, msg, 27)
        self.assertRaises(ValueError, a.send_bytes, msg, 22, 5)
        self.assertRaises(ValueError, a.send_bytes, msg, 26, 1)
        self.assertRaises(ValueError, a.send_bytes, msg, -1)
        self.assertRaises(ValueError, a.send_bytes, msg, 4, -1)

    @classmethod
    def _is_fd_assigned(cls, fd):
        if False:
            return 10
        try:
            os.fstat(fd)
        except OSError as e:
            if e.errno == errno.EBADF:
                return False
            raise
        else:
            return True

    @classmethod
    def _writefd(cls, conn, data, create_dummy_fds=False):
        if False:
            return 10
        if create_dummy_fds:
            for i in range(0, 256):
                if not cls._is_fd_assigned(i):
                    os.dup2(conn.fileno(), i)
        fd = reduction.recv_handle(conn)
        if msvcrt:
            fd = msvcrt.open_osfhandle(fd, os.O_WRONLY)
        os.write(fd, data)
        os.close(fd)

    @unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
    def test_fd_transfer(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE != 'processes':
            self.skipTest('only makes sense with processes')
        (conn, child_conn) = self.Pipe(duplex=True)
        p = self.Process(target=self._writefd, args=(child_conn, b'foo'))
        p.daemon = True
        p.start()
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with open(os_helper.TESTFN, 'wb') as f:
            fd = f.fileno()
            if msvcrt:
                fd = msvcrt.get_osfhandle(fd)
            reduction.send_handle(conn, fd, p.pid)
        p.join()
        with open(os_helper.TESTFN, 'rb') as f:
            self.assertEqual(f.read(), b'foo')

    @unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
    @unittest.skipIf(sys.platform == 'win32', "test semantics don't make sense on Windows")
    @unittest.skipIf(MAXFD <= 256, 'largest assignable fd number is too small')
    @unittest.skipUnless(hasattr(os, 'dup2'), 'test needs os.dup2()')
    def test_large_fd_transfer(self):
        if False:
            while True:
                i = 10
        if self.TYPE != 'processes':
            self.skipTest('only makes sense with processes')
        (conn, child_conn) = self.Pipe(duplex=True)
        p = self.Process(target=self._writefd, args=(child_conn, b'bar', True))
        p.daemon = True
        p.start()
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with open(os_helper.TESTFN, 'wb') as f:
            fd = f.fileno()
            for newfd in range(256, MAXFD):
                if not self._is_fd_assigned(newfd):
                    break
            else:
                self.fail('could not find an unassigned large file descriptor')
            os.dup2(fd, newfd)
            try:
                reduction.send_handle(conn, newfd, p.pid)
            finally:
                os.close(newfd)
        p.join()
        with open(os_helper.TESTFN, 'rb') as f:
            self.assertEqual(f.read(), b'bar')

    @classmethod
    def _send_data_without_fd(self, conn):
        if False:
            i = 10
            return i + 15
        os.write(conn.fileno(), b'\x00')

    @unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
    @unittest.skipIf(sys.platform == 'win32', "doesn't make sense on Windows")
    def test_missing_fd_transfer(self):
        if False:
            for i in range(10):
                print('nop')
        if self.TYPE != 'processes':
            self.skipTest('only makes sense with processes')
        (conn, child_conn) = self.Pipe(duplex=True)
        p = self.Process(target=self._send_data_without_fd, args=(child_conn,))
        p.daemon = True
        p.start()
        self.assertRaises(RuntimeError, reduction.recv_handle, conn)
        p.join()

    def test_context(self):
        if False:
            print('Hello World!')
        (a, b) = self.Pipe()
        with a, b:
            a.send(1729)
            self.assertEqual(b.recv(), 1729)
            if self.TYPE == 'processes':
                self.assertFalse(a.closed)
                self.assertFalse(b.closed)
        if self.TYPE == 'processes':
            self.assertTrue(a.closed)
            self.assertTrue(b.closed)
            self.assertRaises(OSError, a.recv)
            self.assertRaises(OSError, b.recv)

class _TestListener(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_multiple_bind(self):
        if False:
            i = 10
            return i + 15
        for family in self.connection.families:
            l = self.connection.Listener(family=family)
            self.addCleanup(l.close)
            self.assertRaises(OSError, self.connection.Listener, l.address, family)

    def test_context(self):
        if False:
            print('Hello World!')
        with self.connection.Listener() as l:
            with self.connection.Client(l.address) as c:
                with l.accept() as d:
                    c.send(1729)
                    self.assertEqual(d.recv(), 1729)
        if self.TYPE == 'processes':
            self.assertRaises(OSError, l.accept)

    @unittest.skipUnless(util.abstract_sockets_supported, 'test needs abstract socket support')
    def test_abstract_socket(self):
        if False:
            return 10
        with self.connection.Listener('\x00something') as listener:
            with self.connection.Client(listener.address) as client:
                with listener.accept() as d:
                    client.send(1729)
                    self.assertEqual(d.recv(), 1729)
        if self.TYPE == 'processes':
            self.assertRaises(OSError, listener.accept)

class _TestListenerClient(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    @classmethod
    def _test(cls, address):
        if False:
            print('Hello World!')
        conn = cls.connection.Client(address)
        conn.send('hello')
        conn.close()

    def test_listener_client(self):
        if False:
            return 10
        for family in self.connection.families:
            l = self.connection.Listener(family=family)
            p = self.Process(target=self._test, args=(l.address,))
            p.daemon = True
            p.start()
            conn = l.accept()
            self.assertEqual(conn.recv(), 'hello')
            p.join()
            l.close()

    def test_issue14725(self):
        if False:
            while True:
                i = 10
        l = self.connection.Listener()
        p = self.Process(target=self._test, args=(l.address,))
        p.daemon = True
        p.start()
        time.sleep(1)
        conn = l.accept()
        self.assertEqual(conn.recv(), 'hello')
        conn.close()
        p.join()
        l.close()

    def test_issue16955(self):
        if False:
            return 10
        for fam in self.connection.families:
            l = self.connection.Listener(family=fam)
            c = self.connection.Client(l.address)
            a = l.accept()
            a.send_bytes(b'hello')
            self.assertTrue(c.poll(1))
            a.close()
            c.close()
            l.close()

class _TestPoll(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    def test_empty_string(self):
        if False:
            return 10
        (a, b) = self.Pipe()
        self.assertEqual(a.poll(), False)
        b.send_bytes(b'')
        self.assertEqual(a.poll(), True)
        self.assertEqual(a.poll(), True)

    @classmethod
    def _child_strings(cls, conn, strings):
        if False:
            for i in range(10):
                print('nop')
        for s in strings:
            time.sleep(0.1)
            conn.send_bytes(s)
        conn.close()

    def test_strings(self):
        if False:
            print('Hello World!')
        strings = (b'hello', b'', b'a', b'b', b'', b'bye', b'', b'lop')
        (a, b) = self.Pipe()
        p = self.Process(target=self._child_strings, args=(b, strings))
        p.start()
        for s in strings:
            for i in range(200):
                if a.poll(0.01):
                    break
            x = a.recv_bytes()
            self.assertEqual(s, x)
        p.join()

    @classmethod
    def _child_boundaries(cls, r):
        if False:
            for i in range(10):
                print('nop')
        r.poll(5)

    def test_boundaries(self):
        if False:
            i = 10
            return i + 15
        (r, w) = self.Pipe(False)
        p = self.Process(target=self._child_boundaries, args=(r,))
        p.start()
        time.sleep(2)
        L = [b'first', b'second']
        for obj in L:
            w.send_bytes(obj)
        w.close()
        p.join()
        self.assertIn(r.recv_bytes(), L)

    @classmethod
    def _child_dont_merge(cls, b):
        if False:
            return 10
        b.send_bytes(b'a')
        b.send_bytes(b'b')
        b.send_bytes(b'cd')

    def test_dont_merge(self):
        if False:
            print('Hello World!')
        (a, b) = self.Pipe()
        self.assertEqual(a.poll(0.0), False)
        self.assertEqual(a.poll(0.1), False)
        p = self.Process(target=self._child_dont_merge, args=(b,))
        p.start()
        self.assertEqual(a.recv_bytes(), b'a')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.recv_bytes(), b'b')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(0.0), True)
        self.assertEqual(a.recv_bytes(), b'cd')
        p.join()

@unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
@hashlib_helper.requires_hashdigest('md5')
class _TestPicklingConnections(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        from multiprocessing import resource_sharer
        resource_sharer.stop(timeout=support.LONG_TIMEOUT)

    @classmethod
    def _listener(cls, conn, families):
        if False:
            i = 10
            return i + 15
        for fam in families:
            l = cls.connection.Listener(family=fam)
            conn.send(l.address)
            new_conn = l.accept()
            conn.send(new_conn)
            new_conn.close()
            l.close()
        l = socket.create_server((socket_helper.HOST, 0))
        conn.send(l.getsockname())
        (new_conn, addr) = l.accept()
        conn.send(new_conn)
        new_conn.close()
        l.close()
        conn.recv()

    @classmethod
    def _remote(cls, conn):
        if False:
            i = 10
            return i + 15
        for (address, msg) in iter(conn.recv, None):
            client = cls.connection.Client(address)
            client.send(msg.upper())
            client.close()
        (address, msg) = conn.recv()
        client = socket.socket()
        client.connect(address)
        client.sendall(msg.upper())
        client.close()
        conn.close()

    def test_pickling(self):
        if False:
            while True:
                i = 10
        families = self.connection.families
        (lconn, lconn0) = self.Pipe()
        lp = self.Process(target=self._listener, args=(lconn0, families))
        lp.daemon = True
        lp.start()
        lconn0.close()
        (rconn, rconn0) = self.Pipe()
        rp = self.Process(target=self._remote, args=(rconn0,))
        rp.daemon = True
        rp.start()
        rconn0.close()
        for fam in families:
            msg = ('This connection uses family %s' % fam).encode('ascii')
            address = lconn.recv()
            rconn.send((address, msg))
            new_conn = lconn.recv()
            self.assertEqual(new_conn.recv(), msg.upper())
        rconn.send(None)
        msg = latin('This connection uses a normal socket')
        address = lconn.recv()
        rconn.send((address, msg))
        new_conn = lconn.recv()
        buf = []
        while True:
            s = new_conn.recv(100)
            if not s:
                break
            buf.append(s)
        buf = b''.join(buf)
        self.assertEqual(buf, msg.upper())
        new_conn.close()
        lconn.send(None)
        rconn.close()
        lconn.close()
        lp.join()
        rp.join()

    @classmethod
    def child_access(cls, conn):
        if False:
            while True:
                i = 10
        w = conn.recv()
        w.send('all is well')
        w.close()
        r = conn.recv()
        msg = r.recv()
        conn.send(msg * 2)
        conn.close()

    def test_access(self):
        if False:
            return 10
        (conn, child_conn) = self.Pipe()
        p = self.Process(target=self.child_access, args=(child_conn,))
        p.daemon = True
        p.start()
        child_conn.close()
        (r, w) = self.Pipe(duplex=False)
        conn.send(w)
        w.close()
        self.assertEqual(r.recv(), 'all is well')
        r.close()
        (r, w) = self.Pipe(duplex=False)
        conn.send(r)
        r.close()
        w.send('foobar')
        w.close()
        self.assertEqual(conn.recv(), 'foobar' * 2)
        p.join()

class _TestHeap(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.old_heap = multiprocessing.heap.BufferWrapper._heap
        multiprocessing.heap.BufferWrapper._heap = multiprocessing.heap.Heap()

    def tearDown(self):
        if False:
            print('Hello World!')
        multiprocessing.heap.BufferWrapper._heap = self.old_heap
        super().tearDown()

    def test_heap(self):
        if False:
            print('Hello World!')
        iterations = 5000
        maxblocks = 50
        blocks = []
        heap = multiprocessing.heap.BufferWrapper._heap
        heap._DISCARD_FREE_SPACE_LARGER_THAN = 0
        for i in range(iterations):
            size = int(random.lognormvariate(0, 1) * 1000)
            b = multiprocessing.heap.BufferWrapper(size)
            blocks.append(b)
            if len(blocks) > maxblocks:
                i = random.randrange(maxblocks)
                del blocks[i]
            del b
        with heap._lock:
            all = []
            free = 0
            occupied = 0
            for L in list(heap._len_to_seq.values()):
                for (arena, start, stop) in L:
                    all.append((heap._arenas.index(arena), start, stop, stop - start, 'free'))
                    free += stop - start
            for (arena, arena_blocks) in heap._allocated_blocks.items():
                for (start, stop) in arena_blocks:
                    all.append((heap._arenas.index(arena), start, stop, stop - start, 'occupied'))
                    occupied += stop - start
            self.assertEqual(free + occupied, sum((arena.size for arena in heap._arenas)))
            all.sort()
            for i in range(len(all) - 1):
                (arena, start, stop) = all[i][:3]
                (narena, nstart, nstop) = all[i + 1][:3]
                if arena != narena:
                    self.assertEqual(stop, heap._arenas[arena].size)
                    self.assertEqual(nstart, 0)
                else:
                    self.assertEqual(stop, nstart)
        random.shuffle(blocks)
        while blocks:
            blocks.pop()
        self.assertEqual(heap._n_frees, heap._n_mallocs)
        self.assertEqual(len(heap._pending_free_blocks), 0)
        self.assertEqual(len(heap._arenas), 0)
        self.assertEqual(len(heap._allocated_blocks), 0, heap._allocated_blocks)
        self.assertEqual(len(heap._len_to_seq), 0)

    def test_free_from_gc(self):
        if False:
            print('Hello World!')
        if not gc.isenabled():
            gc.enable()
            self.addCleanup(gc.disable)
        thresholds = gc.get_threshold()
        self.addCleanup(gc.set_threshold, *thresholds)
        gc.set_threshold(10)
        for i in range(5000):
            a = multiprocessing.heap.BufferWrapper(1)
            b = multiprocessing.heap.BufferWrapper(1)
            a.buddy = b
            b.buddy = a

class _Foo(Structure):
    _fields_ = [('x', c_int), ('y', c_double), ('z', c_longlong)]

class _TestSharedCTypes(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        if False:
            while True:
                i = 10
        if not HAS_SHAREDCTYPES:
            self.skipTest('requires multiprocessing.sharedctypes')

    @classmethod
    def _double(cls, x, y, z, foo, arr, string):
        if False:
            i = 10
            return i + 15
        x.value *= 2
        y.value *= 2
        z.value *= 2
        foo.x *= 2
        foo.y *= 2
        string.value *= 2
        for i in range(len(arr)):
            arr[i] *= 2

    def test_sharedctypes(self, lock=False):
        if False:
            while True:
                i = 10
        x = Value('i', 7, lock=lock)
        y = Value(c_double, 1.0 / 3.0, lock=lock)
        z = Value(c_longlong, 2 ** 33, lock=lock)
        foo = Value(_Foo, 3, 2, lock=lock)
        arr = self.Array('d', list(range(10)), lock=lock)
        string = self.Array('c', 20, lock=lock)
        string.value = latin('hello')
        p = self.Process(target=self._double, args=(x, y, z, foo, arr, string))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(x.value, 14)
        self.assertAlmostEqual(y.value, 2.0 / 3.0)
        self.assertEqual(z.value, 2 ** 34)
        self.assertEqual(foo.x, 6)
        self.assertAlmostEqual(foo.y, 4.0)
        for i in range(10):
            self.assertAlmostEqual(arr[i], i * 2)
        self.assertEqual(string.value, latin('hellohello'))

    def test_synchronize(self):
        if False:
            i = 10
            return i + 15
        self.test_sharedctypes(lock=True)

    def test_copy(self):
        if False:
            return 10
        foo = _Foo(2, 5.0, 2 ** 33)
        bar = copy(foo)
        foo.x = 0
        foo.y = 0
        foo.z = 0
        self.assertEqual(bar.x, 2)
        self.assertAlmostEqual(bar.y, 5.0)
        self.assertEqual(bar.z, 2 ** 33)

@unittest.skipUnless(HAS_SHMEM, 'requires multiprocessing.shared_memory')
@hashlib_helper.requires_hashdigest('md5')
class _TestSharedMemory(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @staticmethod
    def _attach_existing_shmem_then_write(shmem_name_or_obj, binary_data):
        if False:
            print('Hello World!')
        if isinstance(shmem_name_or_obj, str):
            local_sms = shared_memory.SharedMemory(shmem_name_or_obj)
        else:
            local_sms = shmem_name_or_obj
        local_sms.buf[:len(binary_data)] = binary_data
        local_sms.close()

    def _new_shm_name(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        return prefix + str(os.getpid())

    def test_shared_memory_basics(self):
        if False:
            i = 10
            return i + 15
        name_tsmb = self._new_shm_name('test01_tsmb')
        sms = shared_memory.SharedMemory(name_tsmb, create=True, size=512)
        self.addCleanup(sms.unlink)
        self.assertEqual(sms.name, name_tsmb)
        self.assertGreaterEqual(sms.size, 512)
        self.assertGreaterEqual(len(sms.buf), sms.size)
        self.assertIn(sms.name, str(sms))
        self.assertIn(str(sms.size), str(sms))
        sms.buf[0] = 42
        self.assertEqual(sms.buf[0], 42)
        also_sms = shared_memory.SharedMemory(name_tsmb)
        self.assertEqual(also_sms.buf[0], 42)
        also_sms.close()
        same_sms = shared_memory.SharedMemory(name_tsmb, size=20 * sms.size)
        self.assertLess(same_sms.size, 20 * sms.size)
        same_sms.close()
        with self.assertRaises(ValueError):
            shared_memory.SharedMemory(create=True, size=-2)
        with self.assertRaises(ValueError):
            shared_memory.SharedMemory(create=False)
        with unittest.mock.patch('multiprocessing.shared_memory._make_filename') as mock_make_filename:
            NAME_PREFIX = shared_memory._SHM_NAME_PREFIX
            names = [self._new_shm_name('test01_fn'), self._new_shm_name('test02_fn')]
            names = [NAME_PREFIX + name for name in names]
            mock_make_filename.side_effect = names
            shm1 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm1.unlink)
            self.assertEqual(shm1._name, names[0])
            mock_make_filename.side_effect = names
            shm2 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm2.unlink)
            self.assertEqual(shm2._name, names[1])
        if shared_memory._USE_POSIX:
            name_dblunlink = self._new_shm_name('test01_dblunlink')
            sms_uno = shared_memory.SharedMemory(name_dblunlink, create=True, size=5000)
            with self.assertRaises(FileNotFoundError):
                try:
                    self.assertGreaterEqual(sms_uno.size, 5000)
                    sms_duo = shared_memory.SharedMemory(name_dblunlink)
                    sms_duo.unlink()
                    sms_duo.close()
                    sms_uno.close()
                finally:
                    sms_uno.unlink()
        with self.assertRaises(FileExistsError):
            there_can_only_be_one_sms = shared_memory.SharedMemory(name_tsmb, create=True, size=512)
        if shared_memory._USE_POSIX:

            class OptionalAttachSharedMemory(shared_memory.SharedMemory):
                _flags = os.O_CREAT | os.O_RDWR
            ok_if_exists_sms = OptionalAttachSharedMemory(name_tsmb)
            self.assertEqual(ok_if_exists_sms.size, sms.size)
            ok_if_exists_sms.close()
        with self.assertRaises(FileNotFoundError):
            nonexisting_sms = shared_memory.SharedMemory('test01_notthere')
            nonexisting_sms.unlink()
        sms.close()

    def test_shared_memory_recreate(self):
        if False:
            return 10
        with unittest.mock.patch('multiprocessing.shared_memory._make_filename') as mock_make_filename:
            NAME_PREFIX = shared_memory._SHM_NAME_PREFIX
            names = ['test01_fn', 'test02_fn']
            names = [NAME_PREFIX + name for name in names]
            mock_make_filename.side_effect = names
            shm1 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm1.unlink)
            self.assertEqual(shm1._name, names[0])
            mock_make_filename.side_effect = names
            shm2 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm2.unlink)
            self.assertEqual(shm2._name, names[1])

    def test_invalid_shared_memory_cration(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True, size=-1)
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True, size=0)
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True)

    def test_shared_memory_pickle_unpickle(self):
        if False:
            for i in range(10):
                print('nop')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sms = shared_memory.SharedMemory(create=True, size=512)
                self.addCleanup(sms.unlink)
                sms.buf[0:6] = b'pickle'
                pickled_sms = pickle.dumps(sms, protocol=proto)
                sms2 = pickle.loads(pickled_sms)
                self.assertIsInstance(sms2, shared_memory.SharedMemory)
                self.assertEqual(sms.name, sms2.name)
                self.assertEqual(bytes(sms.buf[0:6]), b'pickle')
                self.assertEqual(bytes(sms2.buf[0:6]), b'pickle')
                sms.buf[0:6] = b'newval'
                self.assertEqual(bytes(sms.buf[0:6]), b'newval')
                self.assertEqual(bytes(sms2.buf[0:6]), b'newval')
                sms2.buf[0:6] = b'oldval'
                self.assertEqual(bytes(sms.buf[0:6]), b'oldval')
                self.assertEqual(bytes(sms2.buf[0:6]), b'oldval')

    def test_shared_memory_pickle_unpickle_dead_object(self):
        if False:
            while True:
                i = 10
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sms = shared_memory.SharedMemory(create=True, size=512)
                sms.buf[0:6] = b'pickle'
                pickled_sms = pickle.dumps(sms, protocol=proto)
                sms.close()
                sms.unlink()
                with self.assertRaises(FileNotFoundError):
                    pickle.loads(pickled_sms)

    def test_shared_memory_across_processes(self):
        if False:
            i = 10
            return i + 15
        sms = shared_memory.SharedMemory(create=True, size=512)
        self.addCleanup(sms.unlink)
        p = self.Process(target=self._attach_existing_shmem_then_write, args=(sms.name, b'howdy'))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(bytes(sms.buf[:5]), b'howdy')
        p = self.Process(target=self._attach_existing_shmem_then_write, args=(sms, b'HELLO'))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(bytes(sms.buf[:5]), b'HELLO')
        sms.close()

    @unittest.skipIf(os.name != 'posix', 'not feasible in non-posix platforms')
    def test_shared_memory_SharedMemoryServer_ignores_sigint(self):
        if False:
            while True:
                i = 10
        smm = multiprocessing.managers.SharedMemoryManager()
        smm.start()
        sl = smm.ShareableList(range(10))
        os.kill(smm._process.pid, signal.SIGINT)
        sl2 = smm.ShareableList(range(10))
        with self.assertRaises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
        smm.shutdown()

    @unittest.skipIf(os.name != 'posix', 'resource_tracker is posix only')
    def test_shared_memory_SharedMemoryManager_reuses_resource_tracker(self):
        if False:
            i = 10
            return i + 15
        cmd = 'if 1:\n            from multiprocessing.managers import SharedMemoryManager\n\n\n            smm = SharedMemoryManager()\n            smm.start()\n            sl = smm.ShareableList(range(10))\n            smm.shutdown()\n        '
        (rc, out, err) = test.support.script_helper.assert_python_ok('-c', cmd)
        self.assertFalse(err)

    def test_shared_memory_SharedMemoryManager_basics(self):
        if False:
            for i in range(10):
                print('nop')
        smm1 = multiprocessing.managers.SharedMemoryManager()
        with self.assertRaises(ValueError):
            smm1.SharedMemory(size=9)
        smm1.start()
        lol = [smm1.ShareableList(range(i)) for i in range(5, 10)]
        lom = [smm1.SharedMemory(size=j) for j in range(32, 128, 16)]
        doppleganger_list0 = shared_memory.ShareableList(name=lol[0].shm.name)
        self.assertEqual(len(doppleganger_list0), 5)
        doppleganger_shm0 = shared_memory.SharedMemory(name=lom[0].name)
        self.assertGreaterEqual(len(doppleganger_shm0.buf), 32)
        held_name = lom[0].name
        smm1.shutdown()
        if sys.platform != 'win32':
            with self.assertRaises(FileNotFoundError):
                absent_shm = shared_memory.SharedMemory(name=held_name)
        with multiprocessing.managers.SharedMemoryManager() as smm2:
            sl = smm2.ShareableList('howdy')
            shm = smm2.SharedMemory(size=128)
            held_name = sl.shm.name
        if sys.platform != 'win32':
            with self.assertRaises(FileNotFoundError):
                absent_sl = shared_memory.ShareableList(name=held_name)

    def test_shared_memory_ShareableList_basics(self):
        if False:
            for i in range(10):
                print('nop')
        sl = shared_memory.ShareableList(['howdy', b'HoWdY', -273.154, 100, None, True, 42])
        self.addCleanup(sl.shm.unlink)
        self.assertIn(sl.shm.name, str(sl))
        self.assertIn(str(list(sl)), str(sl))
        with self.assertRaises(IndexError):
            sl[7]
        with self.assertRaises(IndexError):
            sl[7] = 2
        current_format = sl._get_packing_format(0)
        sl[0] = 'howdy'
        self.assertEqual(current_format, sl._get_packing_format(0))
        self.assertEqual(sl.format, '8s8sdqxxxxxx?xxxxxxxx?q')
        self.assertEqual(len(sl), 7)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                sl.index('100')
            self.assertEqual(sl.index(100), 3)
        self.assertEqual(sl[0], 'howdy')
        self.assertEqual(sl[-2], True)
        self.assertEqual(tuple(sl), ('howdy', b'HoWdY', -273.154, 100, None, True, 42))
        sl[3] = 42
        self.assertEqual(sl[3], 42)
        sl[4] = 'some'
        self.assertEqual(sl[4], 'some')
        self.assertEqual(sl.format, '8s8sdq8sxxxxxxx?q')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[4] = 'far too many'
        self.assertEqual(sl[4], 'some')
        sl[0] = 'encodés'
        self.assertEqual(sl[0], 'encodés')
        self.assertEqual(sl[1], b'HoWdY')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[0] = 'encodées'
        self.assertEqual(sl[1], b'HoWdY')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[1] = b'123456789'
        self.assertEqual(sl[1], b'HoWdY')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(sl.count(42), 2)
            self.assertEqual(sl.count(b'HoWdY'), 1)
            self.assertEqual(sl.count(b'adios'), 0)
        name_duplicate = self._new_shm_name('test03_duplicate')
        sl_copy = shared_memory.ShareableList(sl, name=name_duplicate)
        try:
            self.assertNotEqual(sl.shm.name, sl_copy.shm.name)
            self.assertEqual(name_duplicate, sl_copy.shm.name)
            self.assertEqual(list(sl), list(sl_copy))
            self.assertEqual(sl.format, sl_copy.format)
            sl_copy[-1] = 77
            self.assertEqual(sl_copy[-1], 77)
            self.assertNotEqual(sl[-1], 77)
            sl_copy.shm.close()
        finally:
            sl_copy.shm.unlink()
        sl_tethered = shared_memory.ShareableList(name=sl.shm.name)
        self.assertEqual(sl.shm.name, sl_tethered.shm.name)
        sl_tethered[-1] = 880
        self.assertEqual(sl[-1], 880)
        sl_tethered.shm.close()
        sl.shm.close()
        empty_sl = shared_memory.ShareableList()
        try:
            self.assertEqual(len(empty_sl), 0)
            self.assertEqual(empty_sl.format, '')
            self.assertEqual(empty_sl.count('any'), 0)
            with self.assertRaises(ValueError):
                empty_sl.index(None)
            empty_sl.shm.close()
        finally:
            empty_sl.shm.unlink()

    def test_shared_memory_ShareableList_pickling(self):
        if False:
            print('Hello World!')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sl = shared_memory.ShareableList(range(10))
                self.addCleanup(sl.shm.unlink)
                serialized_sl = pickle.dumps(sl, protocol=proto)
                deserialized_sl = pickle.loads(serialized_sl)
                self.assertIsInstance(deserialized_sl, shared_memory.ShareableList)
                self.assertEqual(deserialized_sl[-1], 9)
                self.assertIsNot(sl, deserialized_sl)
                deserialized_sl[4] = 'changed'
                self.assertEqual(sl[4], 'changed')
                sl[3] = 'newvalue'
                self.assertEqual(deserialized_sl[3], 'newvalue')
                larger_sl = shared_memory.ShareableList(range(400))
                self.addCleanup(larger_sl.shm.unlink)
                serialized_larger_sl = pickle.dumps(larger_sl, protocol=proto)
                self.assertEqual(len(serialized_sl), len(serialized_larger_sl))
                larger_sl.shm.close()
                deserialized_sl.shm.close()
                sl.shm.close()

    def test_shared_memory_ShareableList_pickling_dead_object(self):
        if False:
            i = 10
            return i + 15
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sl = shared_memory.ShareableList(range(10))
                serialized_sl = pickle.dumps(sl, protocol=proto)
                sl.shm.close()
                sl.shm.unlink()
                with self.assertRaises(FileNotFoundError):
                    pickle.loads(serialized_sl)

    def test_shared_memory_cleaned_after_process_termination(self):
        if False:
            i = 10
            return i + 15
        cmd = "if 1:\n            import os, time, sys\n            from multiprocessing import shared_memory\n\n            # Create a shared_memory segment, and send the segment name\n            sm = shared_memory.SharedMemory(create=True, size=10)\n            sys.stdout.write(sm.name + '\\n')\n            sys.stdout.flush()\n            time.sleep(100)\n        "
        with subprocess.Popen([sys.executable, '-E', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            name = p.stdout.readline().strip().decode()
            p.terminate()
            p.wait()
            deadline = time.monotonic() + support.LONG_TIMEOUT
            t = 0.1
            while time.monotonic() < deadline:
                time.sleep(t)
                t = min(t * 2, 5)
                try:
                    smm = shared_memory.SharedMemory(name, create=False)
                except FileNotFoundError:
                    break
            else:
                raise AssertionError('A SharedMemory segment was leaked after a process was abruptly terminated.')
            if os.name == 'posix':
                resource_tracker.unregister(f'/{name}', 'shared_memory')
                err = p.stderr.read().decode()
                self.assertIn('resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown', err)

class _TestFinalize(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        if False:
            return 10
        self.registry_backup = util._finalizer_registry.copy()
        util._finalizer_registry.clear()

    def tearDown(self):
        if False:
            print('Hello World!')
        gc.collect()
        self.assertFalse(util._finalizer_registry)
        util._finalizer_registry.update(self.registry_backup)

    @classmethod
    def _test_finalize(cls, conn):
        if False:
            print('Hello World!')

        class Foo(object):
            pass
        a = Foo()
        util.Finalize(a, conn.send, args=('a',))
        del a
        gc.collect()
        b = Foo()
        close_b = util.Finalize(b, conn.send, args=('b',))
        close_b()
        close_b()
        del b
        gc.collect()
        c = Foo()
        util.Finalize(c, conn.send, args=('c',))
        d10 = Foo()
        util.Finalize(d10, conn.send, args=('d10',), exitpriority=1)
        d01 = Foo()
        util.Finalize(d01, conn.send, args=('d01',), exitpriority=0)
        d02 = Foo()
        util.Finalize(d02, conn.send, args=('d02',), exitpriority=0)
        d03 = Foo()
        util.Finalize(d03, conn.send, args=('d03',), exitpriority=0)
        util.Finalize(None, conn.send, args=('e',), exitpriority=-10)
        util.Finalize(None, conn.send, args=('STOP',), exitpriority=-100)
        util._exit_function()
        conn.close()
        os._exit(0)

    def test_finalize(self):
        if False:
            for i in range(10):
                print('nop')
        (conn, child_conn) = self.Pipe()
        p = self.Process(target=self._test_finalize, args=(child_conn,))
        p.daemon = True
        p.start()
        p.join()
        result = [obj for obj in iter(conn.recv, 'STOP')]
        self.assertEqual(result, ['a', 'b', 'd10', 'd03', 'd02', 'd01', 'e'])

    def test_thread_safety(self):
        if False:
            for i in range(10):
                print('nop')

        def cb():
            if False:
                return 10
            pass

        class Foo(object):

            def __init__(self):
                if False:
                    return 10
                self.ref = self
                util.Finalize(self, cb, exitpriority=random.randint(1, 100))
        finish = False
        exc = None

        def run_finalizers():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal exc
            while not finish:
                time.sleep(random.random() * 0.1)
                try:
                    util._run_finalizers()
                except Exception as e:
                    exc = e

        def make_finalizers():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal exc
            d = {}
            while not finish:
                try:
                    d[random.getrandbits(5)] = {Foo() for i in range(10)}
                except Exception as e:
                    exc = e
                    d.clear()
        old_interval = sys.getswitchinterval()
        old_threshold = gc.get_threshold()
        try:
            sys.setswitchinterval(1e-06)
            gc.set_threshold(5, 5, 5)
            threads = [threading.Thread(target=run_finalizers), threading.Thread(target=make_finalizers)]
            with threading_helper.start_threads(threads):
                time.sleep(4.0)
                finish = True
            if exc is not None:
                raise exc
        finally:
            sys.setswitchinterval(old_interval)
            gc.set_threshold(*old_threshold)
            gc.collect()

class _TestImportStar(unittest.TestCase):

    def get_module_names(self):
        if False:
            print('Hello World!')
        import glob
        folder = os.path.dirname(multiprocessing.__file__)
        pattern = os.path.join(glob.escape(folder), '*.py')
        files = glob.glob(pattern)
        modules = [os.path.splitext(os.path.split(f)[1])[0] for f in files]
        modules = ['multiprocessing.' + m for m in modules]
        modules.remove('multiprocessing.__init__')
        modules.append('multiprocessing')
        return modules

    def test_import(self):
        if False:
            while True:
                i = 10
        modules = self.get_module_names()
        if sys.platform == 'win32':
            modules.remove('multiprocessing.popen_fork')
            modules.remove('multiprocessing.popen_forkserver')
            modules.remove('multiprocessing.popen_spawn_posix')
        else:
            modules.remove('multiprocessing.popen_spawn_win32')
            if not HAS_REDUCTION:
                modules.remove('multiprocessing.popen_forkserver')
        if c_int is None:
            modules.remove('multiprocessing.sharedctypes')
        for name in modules:
            __import__(name)
            mod = sys.modules[name]
            self.assertTrue(hasattr(mod, '__all__'), name)
            for attr in mod.__all__:
                self.assertTrue(hasattr(mod, attr), '%r does not have attribute %r' % (mod, attr))

class _TestLogging(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def test_enable_logging(self):
        if False:
            print('Hello World!')
        logger = multiprocessing.get_logger()
        logger.setLevel(util.SUBWARNING)
        self.assertTrue(logger is not None)
        logger.debug('this will not be printed')
        logger.info('nor will this')
        logger.setLevel(LOG_LEVEL)

    @classmethod
    def _test_level(cls, conn):
        if False:
            print('Hello World!')
        logger = multiprocessing.get_logger()
        conn.send(logger.getEffectiveLevel())

    def test_level(self):
        if False:
            i = 10
            return i + 15
        LEVEL1 = 32
        LEVEL2 = 37
        logger = multiprocessing.get_logger()
        root_logger = logging.getLogger()
        root_level = root_logger.level
        (reader, writer) = multiprocessing.Pipe(duplex=False)
        logger.setLevel(LEVEL1)
        p = self.Process(target=self._test_level, args=(writer,))
        p.start()
        self.assertEqual(LEVEL1, reader.recv())
        p.join()
        p.close()
        logger.setLevel(logging.NOTSET)
        root_logger.setLevel(LEVEL2)
        p = self.Process(target=self._test_level, args=(writer,))
        p.start()
        self.assertEqual(LEVEL2, reader.recv())
        p.join()
        p.close()
        root_logger.setLevel(root_level)
        logger.setLevel(level=LOG_LEVEL)

class _TestPollEintr(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def _killer(cls, pid):
        if False:
            print('Hello World!')
        time.sleep(0.1)
        os.kill(pid, signal.SIGUSR1)

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_poll_eintr(self):
        if False:
            print('Hello World!')
        got_signal = [False]

        def record(*args):
            if False:
                for i in range(10):
                    print('nop')
            got_signal[0] = True
        pid = os.getpid()
        oldhandler = signal.signal(signal.SIGUSR1, record)
        try:
            killer = self.Process(target=self._killer, args=(pid,))
            killer.start()
            try:
                p = self.Process(target=time.sleep, args=(2,))
                p.start()
                p.join()
            finally:
                killer.join()
            self.assertTrue(got_signal[0])
            self.assertEqual(p.exitcode, 0)
        finally:
            signal.signal(signal.SIGUSR1, oldhandler)

class TestInvalidHandle(unittest.TestCase):

    @unittest.skipIf(WIN32, 'skipped on Windows')
    def test_invalid_handles(self):
        if False:
            return 10
        conn = multiprocessing.connection.Connection(44977608)
        try:
            conn.poll()
        except (ValueError, OSError):
            pass
        finally:
            conn._handle = None
        self.assertRaises((ValueError, OSError), multiprocessing.connection.Connection, -1)

@hashlib_helper.requires_hashdigest('md5')
class OtherTest(unittest.TestCase):

    def test_deliver_challenge_auth_failure(self):
        if False:
            print('Hello World!')

        class _FakeConnection(object):

            def recv_bytes(self, size):
                if False:
                    print('Hello World!')
                return b'something bogus'

            def send_bytes(self, data):
                if False:
                    print('Hello World!')
                pass
        self.assertRaises(multiprocessing.AuthenticationError, multiprocessing.connection.deliver_challenge, _FakeConnection(), b'abc')

    def test_answer_challenge_auth_failure(self):
        if False:
            for i in range(10):
                print('nop')

        class _FakeConnection(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.count = 0

            def recv_bytes(self, size):
                if False:
                    i = 10
                    return i + 15
                self.count += 1
                if self.count == 1:
                    return multiprocessing.connection.CHALLENGE
                elif self.count == 2:
                    return b'something bogus'
                return b''

            def send_bytes(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        self.assertRaises(multiprocessing.AuthenticationError, multiprocessing.connection.answer_challenge, _FakeConnection(), b'abc')

def initializer(ns):
    if False:
        print('Hello World!')
    ns.test += 1

@hashlib_helper.requires_hashdigest('md5')
class TestInitializers(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.mgr = multiprocessing.Manager()
        self.ns = self.mgr.Namespace()
        self.ns.test = 0

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.mgr.shutdown()
        self.mgr.join()

    def test_manager_initializer(self):
        if False:
            return 10
        m = multiprocessing.managers.SyncManager()
        self.assertRaises(TypeError, m.start, 1)
        m.start(initializer, (self.ns,))
        self.assertEqual(self.ns.test, 1)
        m.shutdown()
        m.join()

    def test_pool_initializer(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, multiprocessing.Pool, initializer=1)
        p = multiprocessing.Pool(1, initializer, (self.ns,))
        p.close()
        p.join()
        self.assertEqual(self.ns.test, 1)

def _this_sub_process(q):
    if False:
        print('Hello World!')
    try:
        item = q.get(block=False)
    except pyqueue.Empty:
        pass

def _test_process():
    if False:
        while True:
            i = 10
    queue = multiprocessing.Queue()
    subProc = multiprocessing.Process(target=_this_sub_process, args=(queue,))
    subProc.daemon = True
    subProc.start()
    subProc.join()

def _afunc(x):
    if False:
        print('Hello World!')
    return x * x

def pool_in_process():
    if False:
        while True:
            i = 10
    pool = multiprocessing.Pool(processes=4)
    x = pool.map(_afunc, [1, 2, 3, 4, 5, 6, 7])
    pool.close()
    pool.join()

class _file_like(object):

    def __init__(self, delegate):
        if False:
            i = 10
            return i + 15
        self._delegate = delegate
        self._pid = None

    @property
    def cache(self):
        if False:
            i = 10
            return i + 15
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._cache = []
        return self._cache

    def write(self, data):
        if False:
            return 10
        self.cache.append(data)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self._delegate.write(''.join(self.cache))
        self._cache = []

class TestStdinBadfiledescriptor(unittest.TestCase):

    def test_queue_in_process(self):
        if False:
            for i in range(10):
                print('nop')
        proc = multiprocessing.Process(target=_test_process)
        proc.start()
        proc.join()

    def test_pool_in_process(self):
        if False:
            i = 10
            return i + 15
        p = multiprocessing.Process(target=pool_in_process)
        p.start()
        p.join()

    def test_flushing(self):
        if False:
            for i in range(10):
                print('nop')
        sio = io.StringIO()
        flike = _file_like(sio)
        flike.write('foo')
        proc = multiprocessing.Process(target=lambda : flike.flush())
        flike.flush()
        assert sio.getvalue() == 'foo'

class TestWait(unittest.TestCase):

    @classmethod
    def _child_test_wait(cls, w, slow):
        if False:
            while True:
                i = 10
        for i in range(10):
            if slow:
                time.sleep(random.random() * 0.1)
            w.send((i, os.getpid()))
        w.close()

    def test_wait(self, slow=False):
        if False:
            return 10
        from multiprocessing.connection import wait
        readers = []
        procs = []
        messages = []
        for i in range(4):
            (r, w) = multiprocessing.Pipe(duplex=False)
            p = multiprocessing.Process(target=self._child_test_wait, args=(w, slow))
            p.daemon = True
            p.start()
            w.close()
            readers.append(r)
            procs.append(p)
            self.addCleanup(p.join)
        while readers:
            for r in wait(readers):
                try:
                    msg = r.recv()
                except EOFError:
                    readers.remove(r)
                    r.close()
                else:
                    messages.append(msg)
        messages.sort()
        expected = sorted(((i, p.pid) for i in range(10) for p in procs))
        self.assertEqual(messages, expected)

    @classmethod
    def _child_test_wait_socket(cls, address, slow):
        if False:
            i = 10
            return i + 15
        s = socket.socket()
        s.connect(address)
        for i in range(10):
            if slow:
                time.sleep(random.random() * 0.1)
            s.sendall(('%s\n' % i).encode('ascii'))
        s.close()

    def test_wait_socket(self, slow=False):
        if False:
            i = 10
            return i + 15
        from multiprocessing.connection import wait
        l = socket.create_server((socket_helper.HOST, 0))
        addr = l.getsockname()
        readers = []
        procs = []
        dic = {}
        for i in range(4):
            p = multiprocessing.Process(target=self._child_test_wait_socket, args=(addr, slow))
            p.daemon = True
            p.start()
            procs.append(p)
            self.addCleanup(p.join)
        for i in range(4):
            (r, _) = l.accept()
            readers.append(r)
            dic[r] = []
        l.close()
        while readers:
            for r in wait(readers):
                msg = r.recv(32)
                if not msg:
                    readers.remove(r)
                    r.close()
                else:
                    dic[r].append(msg)
        expected = ''.join(('%s\n' % i for i in range(10))).encode('ascii')
        for v in dic.values():
            self.assertEqual(b''.join(v), expected)

    def test_wait_slow(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_wait(True)

    def test_wait_socket_slow(self):
        if False:
            print('Hello World!')
        self.test_wait_socket(True)

    def test_wait_timeout(self):
        if False:
            while True:
                i = 10
        from multiprocessing.connection import wait
        expected = 5
        (a, b) = multiprocessing.Pipe()
        start = time.monotonic()
        res = wait([a, b], expected)
        delta = time.monotonic() - start
        self.assertEqual(res, [])
        self.assertLess(delta, expected * 2)
        self.assertGreater(delta, expected * 0.5)
        b.send(None)
        start = time.monotonic()
        res = wait([a, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(res, [a])
        self.assertLess(delta, 0.4)

    @classmethod
    def signal_and_sleep(cls, sem, period):
        if False:
            return 10
        sem.release()
        time.sleep(period)

    def test_wait_integer(self):
        if False:
            while True:
                i = 10
        from multiprocessing.connection import wait
        expected = 3
        sorted_ = lambda l: sorted(l, key=lambda x: id(x))
        sem = multiprocessing.Semaphore(0)
        (a, b) = multiprocessing.Pipe()
        p = multiprocessing.Process(target=self.signal_and_sleep, args=(sem, expected))
        p.start()
        self.assertIsInstance(p.sentinel, int)
        self.assertTrue(sem.acquire(timeout=20))
        start = time.monotonic()
        res = wait([a, p.sentinel, b], expected + 20)
        delta = time.monotonic() - start
        self.assertEqual(res, [p.sentinel])
        self.assertLess(delta, expected + 2)
        self.assertGreater(delta, expected - 2)
        a.send(None)
        start = time.monotonic()
        res = wait([a, p.sentinel, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(sorted_(res), sorted_([p.sentinel, b]))
        self.assertLess(delta, 0.4)
        b.send(None)
        start = time.monotonic()
        res = wait([a, p.sentinel, b], 20)
        delta = time.monotonic() - start
        self.assertEqual(sorted_(res), sorted_([a, p.sentinel, b]))
        self.assertLess(delta, 0.4)
        p.terminate()
        p.join()

    def test_neg_timeout(self):
        if False:
            i = 10
            return i + 15
        from multiprocessing.connection import wait
        (a, b) = multiprocessing.Pipe()
        t = time.monotonic()
        res = wait([a], timeout=-1)
        t = time.monotonic() - t
        self.assertEqual(res, [])
        self.assertLess(t, 1)
        a.close()
        b.close()

class TestInvalidFamily(unittest.TestCase):

    @unittest.skipIf(WIN32, 'skipped on Windows')
    def test_invalid_family(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            multiprocessing.connection.Listener('\\\\.\\test')

    @unittest.skipUnless(WIN32, 'skipped on non-Windows platforms')
    def test_invalid_family_win32(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            multiprocessing.connection.Listener('/var/test.pipe')

class TestFlags(unittest.TestCase):

    @classmethod
    def run_in_grandchild(cls, conn):
        if False:
            while True:
                i = 10
        conn.send(tuple(sys.flags))

    @classmethod
    def run_in_child(cls):
        if False:
            return 10
        import json
        (r, w) = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=cls.run_in_grandchild, args=(w,))
        p.start()
        grandchild_flags = r.recv()
        p.join()
        r.close()
        w.close()
        flags = (tuple(sys.flags), grandchild_flags)
        print(json.dumps(flags))

    def test_flags(self):
        if False:
            print('Hello World!')
        import json
        prog = 'from test._test_multiprocessing import TestFlags; ' + 'TestFlags.run_in_child()'
        data = subprocess.check_output([sys.executable, '-E', '-S', '-O', '-c', prog])
        (child_flags, grandchild_flags) = json.loads(data.decode('ascii'))
        self.assertEqual(child_flags, grandchild_flags)

class TestTimeouts(unittest.TestCase):

    @classmethod
    def _test_timeout(cls, child, address):
        if False:
            print('Hello World!')
        time.sleep(1)
        child.send(123)
        child.close()
        conn = multiprocessing.connection.Client(address)
        conn.send(456)
        conn.close()

    def test_timeout(self):
        if False:
            while True:
                i = 10
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(0.1)
            (parent, child) = multiprocessing.Pipe(duplex=True)
            l = multiprocessing.connection.Listener(family='AF_INET')
            p = multiprocessing.Process(target=self._test_timeout, args=(child, l.address))
            p.start()
            child.close()
            self.assertEqual(parent.recv(), 123)
            parent.close()
            conn = l.accept()
            self.assertEqual(conn.recv(), 456)
            conn.close()
            l.close()
            join_process(p)
        finally:
            socket.setdefaulttimeout(old_timeout)

class TestNoForkBomb(unittest.TestCase):

    def test_noforkbomb(self):
        if False:
            for i in range(10):
                print('nop')
        sm = multiprocessing.get_start_method()
        name = os.path.join(os.path.dirname(__file__), 'mp_fork_bomb.py')
        if sm != 'fork':
            (rc, out, err) = test.support.script_helper.assert_python_failure(name, sm)
            self.assertEqual(out, b'')
            self.assertIn(b'RuntimeError', err)
        else:
            (rc, out, err) = test.support.script_helper.assert_python_ok(name, sm)
            self.assertEqual(out.rstrip(), b'123')
            self.assertEqual(err, b'')

class TestForkAwareThreadLock(unittest.TestCase):

    @classmethod
    def child(cls, n, conn):
        if False:
            for i in range(10):
                print('nop')
        if n > 1:
            p = multiprocessing.Process(target=cls.child, args=(n - 1, conn))
            p.start()
            conn.close()
            join_process(p)
        else:
            conn.send(len(util._afterfork_registry))
        conn.close()

    def test_lock(self):
        if False:
            while True:
                i = 10
        (r, w) = multiprocessing.Pipe(False)
        l = util.ForkAwareThreadLock()
        old_size = len(util._afterfork_registry)
        p = multiprocessing.Process(target=self.child, args=(5, w))
        p.start()
        w.close()
        new_size = r.recv()
        join_process(p)
        self.assertLessEqual(new_size, old_size)

class TestCloseFds(unittest.TestCase):

    def get_high_socket_fd(self):
        if False:
            return 10
        if WIN32:
            return socket.socket().detach()
        else:
            fd = socket.socket().detach()
            to_close = []
            while fd < 50:
                to_close.append(fd)
                fd = os.dup(fd)
            for x in to_close:
                os.close(x)
            return fd

    def close(self, fd):
        if False:
            while True:
                i = 10
        if WIN32:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd).close()
        else:
            os.close(fd)

    @classmethod
    def _test_closefds(cls, conn, fd):
        if False:
            print('Hello World!')
        try:
            s = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
        except Exception as e:
            conn.send(e)
        else:
            s.close()
            conn.send(None)

    def test_closefd(self):
        if False:
            for i in range(10):
                print('nop')
        if not HAS_REDUCTION:
            raise unittest.SkipTest('requires fd pickling')
        (reader, writer) = multiprocessing.Pipe()
        fd = self.get_high_socket_fd()
        try:
            p = multiprocessing.Process(target=self._test_closefds, args=(writer, fd))
            p.start()
            writer.close()
            e = reader.recv()
            join_process(p)
        finally:
            self.close(fd)
            writer.close()
            reader.close()
        if multiprocessing.get_start_method() == 'fork':
            self.assertIs(e, None)
        else:
            WSAENOTSOCK = 10038
            self.assertIsInstance(e, OSError)
            self.assertTrue(e.errno == errno.EBADF or e.winerror == WSAENOTSOCK, e)

class TestIgnoreEINTR(unittest.TestCase):
    CONN_MAX_SIZE = max(support.PIPE_MAX_SIZE, support.SOCK_MAX_SIZE)

    @classmethod
    def _test_ignore(cls, conn):
        if False:
            i = 10
            return i + 15

        def handler(signum, frame):
            if False:
                while True:
                    i = 10
            pass
        signal.signal(signal.SIGUSR1, handler)
        conn.send('ready')
        x = conn.recv()
        conn.send(x)
        conn.send_bytes(b'x' * cls.CONN_MAX_SIZE)

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_ignore(self):
        if False:
            while True:
                i = 10
        (conn, child_conn) = multiprocessing.Pipe()
        try:
            p = multiprocessing.Process(target=self._test_ignore, args=(child_conn,))
            p.daemon = True
            p.start()
            child_conn.close()
            self.assertEqual(conn.recv(), 'ready')
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(0.1)
            conn.send(1234)
            self.assertEqual(conn.recv(), 1234)
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            self.assertEqual(conn.recv_bytes(), b'x' * self.CONN_MAX_SIZE)
            time.sleep(0.1)
            p.join()
        finally:
            conn.close()

    @classmethod
    def _test_ignore_listener(cls, conn):
        if False:
            return 10

        def handler(signum, frame):
            if False:
                return 10
            pass
        signal.signal(signal.SIGUSR1, handler)
        with multiprocessing.connection.Listener() as l:
            conn.send(l.address)
            a = l.accept()
            a.send('welcome')

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_ignore_listener(self):
        if False:
            while True:
                i = 10
        (conn, child_conn) = multiprocessing.Pipe()
        try:
            p = multiprocessing.Process(target=self._test_ignore_listener, args=(child_conn,))
            p.daemon = True
            p.start()
            child_conn.close()
            address = conn.recv()
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(0.1)
            client = multiprocessing.connection.Client(address)
            self.assertEqual(client.recv(), 'welcome')
            p.join()
        finally:
            conn.close()

class TestStartMethod(unittest.TestCase):

    @classmethod
    def _check_context(cls, conn):
        if False:
            while True:
                i = 10
        conn.send(multiprocessing.get_start_method())

    def check_context(self, ctx):
        if False:
            i = 10
            return i + 15
        (r, w) = ctx.Pipe(duplex=False)
        p = ctx.Process(target=self._check_context, args=(w,))
        p.start()
        w.close()
        child_method = r.recv()
        r.close()
        p.join()
        self.assertEqual(child_method, ctx.get_start_method())

    def test_context(self):
        if False:
            return 10
        for method in ('fork', 'spawn', 'forkserver'):
            try:
                ctx = multiprocessing.get_context(method)
            except ValueError:
                continue
            self.assertEqual(ctx.get_start_method(), method)
            self.assertIs(ctx.get_context(), ctx)
            self.assertRaises(ValueError, ctx.set_start_method, 'spawn')
            self.assertRaises(ValueError, ctx.set_start_method, None)
            self.check_context(ctx)

    def test_set_get(self):
        if False:
            while True:
                i = 10
        multiprocessing.set_forkserver_preload(PRELOAD)
        count = 0
        old_method = multiprocessing.get_start_method()
        try:
            for method in ('fork', 'spawn', 'forkserver'):
                try:
                    multiprocessing.set_start_method(method, force=True)
                except ValueError:
                    continue
                self.assertEqual(multiprocessing.get_start_method(), method)
                ctx = multiprocessing.get_context()
                self.assertEqual(ctx.get_start_method(), method)
                self.assertTrue(type(ctx).__name__.lower().startswith(method))
                self.assertTrue(ctx.Process.__name__.lower().startswith(method))
                self.check_context(multiprocessing)
                count += 1
        finally:
            multiprocessing.set_start_method(old_method, force=True)
        self.assertGreaterEqual(count, 1)

    def test_get_all(self):
        if False:
            i = 10
            return i + 15
        methods = multiprocessing.get_all_start_methods()
        if sys.platform == 'win32':
            self.assertEqual(methods, ['spawn'])
        else:
            self.assertTrue(methods == ['fork', 'spawn'] or methods == ['spawn', 'fork'] or methods == ['fork', 'spawn', 'forkserver'] or (methods == ['spawn', 'fork', 'forkserver']))

    def test_preload_resources(self):
        if False:
            while True:
                i = 10
        if multiprocessing.get_start_method() != 'forkserver':
            self.skipTest("test only relevant for 'forkserver' method")
        name = os.path.join(os.path.dirname(__file__), 'mp_preload.py')
        (rc, out, err) = test.support.script_helper.assert_python_ok(name)
        out = out.decode()
        err = err.decode()
        if out.rstrip() != 'ok' or err != '':
            print(out)
            print(err)
            self.fail('failed spawning forkserver or grandchild')

@unittest.skipIf(sys.platform == 'win32', "test semantics don't make sense on Windows")
class TestResourceTracker(unittest.TestCase):

    def test_resource_tracker(self):
        if False:
            return 10
        cmd = 'if 1:\n            import time, os, tempfile\n            import multiprocessing as mp\n            from multiprocessing import resource_tracker\n            from multiprocessing.shared_memory import SharedMemory\n\n            mp.set_start_method("spawn")\n            rand = tempfile._RandomNameSequence()\n\n\n            def create_and_register_resource(rtype):\n                if rtype == "semaphore":\n                    lock = mp.Lock()\n                    return lock, lock._semlock.name\n                elif rtype == "shared_memory":\n                    sm = SharedMemory(create=True, size=10)\n                    return sm, sm._name\n                else:\n                    raise ValueError(\n                        "Resource type {{}} not understood".format(rtype))\n\n\n            resource1, rname1 = create_and_register_resource("{rtype}")\n            resource2, rname2 = create_and_register_resource("{rtype}")\n\n            os.write({w}, rname1.encode("ascii") + b"\\n")\n            os.write({w}, rname2.encode("ascii") + b"\\n")\n\n            time.sleep(10)\n        '
        for rtype in resource_tracker._CLEANUP_FUNCS:
            with self.subTest(rtype=rtype):
                if rtype == 'noop':
                    continue
                (r, w) = os.pipe()
                p = subprocess.Popen([sys.executable, '-E', '-c', cmd.format(w=w, rtype=rtype)], pass_fds=[w], stderr=subprocess.PIPE)
                os.close(w)
                with open(r, 'rb', closefd=True) as f:
                    name1 = f.readline().rstrip().decode('ascii')
                    name2 = f.readline().rstrip().decode('ascii')
                _resource_unlink(name1, rtype)
                p.terminate()
                p.wait()
                deadline = time.monotonic() + support.LONG_TIMEOUT
                while time.monotonic() < deadline:
                    time.sleep(0.5)
                    try:
                        _resource_unlink(name2, rtype)
                    except OSError as e:
                        self.assertIn(e.errno, (errno.ENOENT, errno.EINVAL))
                        break
                else:
                    raise AssertionError(f'A {rtype} resource was leaked after a process was abruptly terminated.')
                err = p.stderr.read().decode('utf-8')
                p.stderr.close()
                expected = 'resource_tracker: There appear to be 2 leaked {} objects'.format(rtype)
                self.assertRegex(err, expected)
                self.assertRegex(err, 'resource_tracker: %r: \\[Errno' % name1)

    def check_resource_tracker_death(self, signum, should_die):
        if False:
            i = 10
            return i + 15
        from multiprocessing.resource_tracker import _resource_tracker
        pid = _resource_tracker._pid
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            support.wait_process(pid, exitcode=-signal.SIGKILL)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _resource_tracker.ensure_running()
        pid = _resource_tracker._pid
        os.kill(pid, signum)
        time.sleep(1.0)
        ctx = multiprocessing.get_context('spawn')
        with warnings.catch_warnings(record=True) as all_warn:
            warnings.simplefilter('always')
            sem = ctx.Semaphore()
            sem.acquire()
            sem.release()
            wr = weakref.ref(sem)
            del sem
            gc.collect()
            self.assertIsNone(wr())
            if should_die:
                self.assertEqual(len(all_warn), 1)
                the_warn = all_warn[0]
                self.assertTrue(issubclass(the_warn.category, UserWarning))
                self.assertTrue('resource_tracker: process died' in str(the_warn.message))
            else:
                self.assertEqual(len(all_warn), 0)

    def test_resource_tracker_sigint(self):
        if False:
            while True:
                i = 10
        self.check_resource_tracker_death(signal.SIGINT, False)

    def test_resource_tracker_sigterm(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_resource_tracker_death(signal.SIGTERM, False)

    def test_resource_tracker_sigkill(self):
        if False:
            while True:
                i = 10
        self.check_resource_tracker_death(signal.SIGKILL, True)

    @staticmethod
    def _is_resource_tracker_reused(conn, pid):
        if False:
            for i in range(10):
                print('nop')
        from multiprocessing.resource_tracker import _resource_tracker
        _resource_tracker.ensure_running()
        reused = _resource_tracker._pid in (None, pid)
        reused &= _resource_tracker._check_alive()
        conn.send(reused)

    def test_resource_tracker_reused(self):
        if False:
            for i in range(10):
                print('nop')
        from multiprocessing.resource_tracker import _resource_tracker
        _resource_tracker.ensure_running()
        pid = _resource_tracker._pid
        (r, w) = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=self._is_resource_tracker_reused, args=(w, pid))
        p.start()
        is_resource_tracker_reused = r.recv()
        p.join()
        w.close()
        r.close()
        self.assertTrue(is_resource_tracker_reused)

class TestSimpleQueue(unittest.TestCase):

    @classmethod
    def _test_empty(cls, queue, child_can_start, parent_can_continue):
        if False:
            i = 10
            return i + 15
        child_can_start.wait()
        try:
            queue.put(queue.empty())
            queue.put(queue.empty())
        finally:
            parent_can_continue.set()

    def test_empty(self):
        if False:
            print('Hello World!')
        queue = multiprocessing.SimpleQueue()
        child_can_start = multiprocessing.Event()
        parent_can_continue = multiprocessing.Event()
        proc = multiprocessing.Process(target=self._test_empty, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertTrue(queue.empty())
        child_can_start.set()
        parent_can_continue.wait()
        self.assertFalse(queue.empty())
        self.assertEqual(queue.get(), True)
        self.assertEqual(queue.get(), False)
        self.assertTrue(queue.empty())
        proc.join()

    def test_close(self):
        if False:
            while True:
                i = 10
        queue = multiprocessing.SimpleQueue()
        queue.close()
        queue.close()

    @test.support.cpython_only
    def test_closed(self):
        if False:
            return 10
        queue = multiprocessing.SimpleQueue()
        queue.close()
        self.assertTrue(queue._reader.closed)
        self.assertTrue(queue._writer.closed)

class TestPoolNotLeakOnFailure(unittest.TestCase):

    def test_release_unused_processes(self):
        if False:
            while True:
                i = 10
        will_fail_in = 3
        forked_processes = []

        class FailingForkProcess:

            def __init__(self, **kwargs):
                if False:
                    while True:
                        i = 10
                self.name = 'Fake Process'
                self.exitcode = None
                self.state = None
                forked_processes.append(self)

            def start(self):
                if False:
                    return 10
                nonlocal will_fail_in
                if will_fail_in <= 0:
                    raise OSError('Manually induced OSError')
                will_fail_in -= 1
                self.state = 'started'

            def terminate(self):
                if False:
                    while True:
                        i = 10
                self.state = 'stopping'

            def join(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self.state == 'stopping':
                    self.state = 'stopped'

            def is_alive(self):
                if False:
                    while True:
                        i = 10
                return self.state == 'started' or self.state == 'stopping'
        with self.assertRaisesRegex(OSError, 'Manually induced OSError'):
            p = multiprocessing.pool.Pool(5, context=unittest.mock.MagicMock(Process=FailingForkProcess))
            p.close()
            p.join()
        self.assertFalse(any((process.is_alive() for process in forked_processes)))

@hashlib_helper.requires_hashdigest('md5')
class TestSyncManagerTypes(unittest.TestCase):
    """Test all the types which can be shared between a parent and a
    child process by using a manager which acts as an intermediary
    between them.

    In the following unit-tests the base type is created in the parent
    process, the @classmethod represents the worker process and the
    shared object is readable and editable between the two.

    # The child.
    @classmethod
    def _test_list(cls, obj):
        assert obj[0] == 5
        assert obj.append(6)

    # The parent.
    def test_list(self):
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert o[1] == 6
    """
    manager_class = multiprocessing.managers.SyncManager

    def setUp(self):
        if False:
            print('Hello World!')
        self.manager = self.manager_class()
        self.manager.start()
        self.proc = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()
        self.manager.shutdown()
        self.manager = None
        self.proc = None

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        support.reap_children()
    tearDownClass = setUpClass

    def wait_proc_exit(self):
        if False:
            for i in range(10):
                print('nop')
        join_process(self.proc)
        start_time = time.monotonic()
        t = 0.01
        while len(multiprocessing.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = time.monotonic() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                support.print_warning(f'multiprocessing.Manager still has {multiprocessing.active_children()} active children after {dt} seconds')
                break

    def run_worker(self, worker, obj):
        if False:
            print('Hello World!')
        self.proc = multiprocessing.Process(target=worker, args=(obj,))
        self.proc.daemon = True
        self.proc.start()
        self.wait_proc_exit()
        self.assertEqual(self.proc.exitcode, 0)

    @classmethod
    def _test_event(cls, obj):
        if False:
            i = 10
            return i + 15
        assert obj.is_set()
        obj.wait()
        obj.clear()
        obj.wait(0.001)

    def test_event(self):
        if False:
            while True:
                i = 10
        o = self.manager.Event()
        o.set()
        self.run_worker(self._test_event, o)
        assert not o.is_set()
        o.wait(0.001)

    @classmethod
    def _test_lock(cls, obj):
        if False:
            while True:
                i = 10
        obj.acquire()

    def test_lock(self, lname='Lock'):
        if False:
            print('Hello World!')
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_lock, o)
        o.release()
        self.assertRaises(RuntimeError, o.release)

    @classmethod
    def _test_rlock(cls, obj):
        if False:
            print('Hello World!')
        obj.acquire()
        obj.release()

    def test_rlock(self, lname='Lock'):
        if False:
            for i in range(10):
                print('nop')
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_rlock, o)

    @classmethod
    def _test_semaphore(cls, obj):
        if False:
            print('Hello World!')
        obj.acquire()

    def test_semaphore(self, sname='Semaphore'):
        if False:
            while True:
                i = 10
        o = getattr(self.manager, sname)()
        self.run_worker(self._test_semaphore, o)
        o.release()

    def test_bounded_semaphore(self):
        if False:
            i = 10
            return i + 15
        self.test_semaphore(sname='BoundedSemaphore')

    @classmethod
    def _test_condition(cls, obj):
        if False:
            for i in range(10):
                print('nop')
        obj.acquire()
        obj.release()

    def test_condition(self):
        if False:
            for i in range(10):
                print('nop')
        o = self.manager.Condition()
        self.run_worker(self._test_condition, o)

    @classmethod
    def _test_barrier(cls, obj):
        if False:
            print('Hello World!')
        assert obj.parties == 5
        obj.reset()

    def test_barrier(self):
        if False:
            while True:
                i = 10
        o = self.manager.Barrier(5)
        self.run_worker(self._test_barrier, o)

    @classmethod
    def _test_pool(cls, obj):
        if False:
            print('Hello World!')
        with obj:
            pass

    def test_pool(self):
        if False:
            while True:
                i = 10
        o = self.manager.Pool(processes=4)
        self.run_worker(self._test_pool, o)

    @classmethod
    def _test_queue(cls, obj):
        if False:
            for i in range(10):
                print('nop')
        assert obj.qsize() == 2
        assert obj.full()
        assert not obj.empty()
        assert obj.get() == 5
        assert not obj.empty()
        assert obj.get() == 6
        assert obj.empty()

    def test_queue(self, qname='Queue'):
        if False:
            print('Hello World!')
        o = getattr(self.manager, qname)(2)
        o.put(5)
        o.put(6)
        self.run_worker(self._test_queue, o)
        assert o.empty()
        assert not o.full()

    def test_joinable_queue(self):
        if False:
            i = 10
            return i + 15
        self.test_queue('JoinableQueue')

    @classmethod
    def _test_list(cls, obj):
        if False:
            while True:
                i = 10
        assert obj[0] == 5
        assert obj.count(5) == 1
        assert obj.index(5) == 0
        obj.sort()
        obj.reverse()
        for x in obj:
            pass
        assert len(obj) == 1
        assert obj.pop(0) == 5

    def test_list(self):
        if False:
            return 10
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_dict(cls, obj):
        if False:
            for i in range(10):
                print('nop')
        assert len(obj) == 1
        assert obj['foo'] == 5
        assert obj.get('foo') == 5
        assert list(obj.items()) == [('foo', 5)]
        assert list(obj.keys()) == ['foo']
        assert list(obj.values()) == [5]
        assert obj.copy() == {'foo': 5}
        assert obj.popitem() == ('foo', 5)

    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')
        o = self.manager.dict()
        o['foo'] = 5
        self.run_worker(self._test_dict, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_value(cls, obj):
        if False:
            return 10
        assert obj.value == 1
        assert obj.get() == 1
        obj.set(2)

    def test_value(self):
        if False:
            i = 10
            return i + 15
        o = self.manager.Value('i', 1)
        self.run_worker(self._test_value, o)
        self.assertEqual(o.value, 2)
        self.assertEqual(o.get(), 2)

    @classmethod
    def _test_array(cls, obj):
        if False:
            print('Hello World!')
        assert obj[0] == 0
        assert obj[1] == 1
        assert len(obj) == 2
        assert list(obj) == [0, 1]

    def test_array(self):
        if False:
            for i in range(10):
                print('nop')
        o = self.manager.Array('i', [0, 1])
        self.run_worker(self._test_array, o)

    @classmethod
    def _test_namespace(cls, obj):
        if False:
            for i in range(10):
                print('nop')
        assert obj.x == 0
        assert obj.y == 1

    def test_namespace(self):
        if False:
            return 10
        o = self.manager.Namespace()
        o.x = 0
        o.y = 1
        self.run_worker(self._test_namespace, o)

class MiscTestCase(unittest.TestCase):

    def test__all__(self):
        if False:
            while True:
                i = 10
        support.check__all__(self, multiprocessing, extra=multiprocessing.__all__, not_exported=['SUBDEBUG', 'SUBWARNING'])

class BaseMixin(object):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.dangling = (multiprocessing.process._dangling.copy(), threading._dangling.copy())

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        test.support.gc_collect()
        processes = set(multiprocessing.process._dangling) - set(cls.dangling[0])
        if processes:
            test.support.environment_altered = True
            support.print_warning(f'Dangling processes: {processes}')
        processes = None
        threads = set(threading._dangling) - set(cls.dangling[1])
        if threads:
            test.support.environment_altered = True
            support.print_warning(f'Dangling threads: {threads}')
        threads = None

class ProcessesMixin(BaseMixin):
    TYPE = 'processes'
    Process = multiprocessing.Process
    connection = multiprocessing.connection
    current_process = staticmethod(multiprocessing.current_process)
    parent_process = staticmethod(multiprocessing.parent_process)
    active_children = staticmethod(multiprocessing.active_children)
    Pool = staticmethod(multiprocessing.Pool)
    Pipe = staticmethod(multiprocessing.Pipe)
    Queue = staticmethod(multiprocessing.Queue)
    JoinableQueue = staticmethod(multiprocessing.JoinableQueue)
    Lock = staticmethod(multiprocessing.Lock)
    RLock = staticmethod(multiprocessing.RLock)
    Semaphore = staticmethod(multiprocessing.Semaphore)
    BoundedSemaphore = staticmethod(multiprocessing.BoundedSemaphore)
    Condition = staticmethod(multiprocessing.Condition)
    Event = staticmethod(multiprocessing.Event)
    Barrier = staticmethod(multiprocessing.Barrier)
    Value = staticmethod(multiprocessing.Value)
    Array = staticmethod(multiprocessing.Array)
    RawValue = staticmethod(multiprocessing.RawValue)
    RawArray = staticmethod(multiprocessing.RawArray)

class ManagerMixin(BaseMixin):
    TYPE = 'manager'
    Process = multiprocessing.Process
    Queue = property(operator.attrgetter('manager.Queue'))
    JoinableQueue = property(operator.attrgetter('manager.JoinableQueue'))
    Lock = property(operator.attrgetter('manager.Lock'))
    RLock = property(operator.attrgetter('manager.RLock'))
    Semaphore = property(operator.attrgetter('manager.Semaphore'))
    BoundedSemaphore = property(operator.attrgetter('manager.BoundedSemaphore'))
    Condition = property(operator.attrgetter('manager.Condition'))
    Event = property(operator.attrgetter('manager.Event'))
    Barrier = property(operator.attrgetter('manager.Barrier'))
    Value = property(operator.attrgetter('manager.Value'))
    Array = property(operator.attrgetter('manager.Array'))
    list = property(operator.attrgetter('manager.list'))
    dict = property(operator.attrgetter('manager.dict'))
    Namespace = property(operator.attrgetter('manager.Namespace'))

    @classmethod
    def Pool(cls, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        return cls.manager.Pool(*args, **kwds)

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.manager = multiprocessing.Manager()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        start_time = time.monotonic()
        t = 0.01
        while len(multiprocessing.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = time.monotonic() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                support.print_warning(f'multiprocessing.Manager still has {multiprocessing.active_children()} active children after {dt} seconds')
                break
        gc.collect()
        if cls.manager._number_of_objects() != 0:
            test.support.environment_altered = True
            support.print_warning('Shared objects which still exist at manager shutdown:')
            support.print_warning(cls.manager._debug_info())
        cls.manager.shutdown()
        cls.manager.join()
        cls.manager = None
        super().tearDownClass()

class ThreadsMixin(BaseMixin):
    TYPE = 'threads'
    Process = multiprocessing.dummy.Process
    connection = multiprocessing.dummy.connection
    current_process = staticmethod(multiprocessing.dummy.current_process)
    active_children = staticmethod(multiprocessing.dummy.active_children)
    Pool = staticmethod(multiprocessing.dummy.Pool)
    Pipe = staticmethod(multiprocessing.dummy.Pipe)
    Queue = staticmethod(multiprocessing.dummy.Queue)
    JoinableQueue = staticmethod(multiprocessing.dummy.JoinableQueue)
    Lock = staticmethod(multiprocessing.dummy.Lock)
    RLock = staticmethod(multiprocessing.dummy.RLock)
    Semaphore = staticmethod(multiprocessing.dummy.Semaphore)
    BoundedSemaphore = staticmethod(multiprocessing.dummy.BoundedSemaphore)
    Condition = staticmethod(multiprocessing.dummy.Condition)
    Event = staticmethod(multiprocessing.dummy.Event)
    Barrier = staticmethod(multiprocessing.dummy.Barrier)
    Value = staticmethod(multiprocessing.dummy.Value)
    Array = staticmethod(multiprocessing.dummy.Array)

def install_tests_in_module_dict(remote_globs, start_method):
    if False:
        for i in range(10):
            print('nop')
    __module__ = remote_globs['__name__']
    local_globs = globals()
    ALL_TYPES = {'processes', 'threads', 'manager'}
    for (name, base) in local_globs.items():
        if not isinstance(base, type):
            continue
        if issubclass(base, BaseTestCase):
            if base is BaseTestCase:
                continue
            assert set(base.ALLOWED_TYPES) <= ALL_TYPES, base.ALLOWED_TYPES
            for type_ in base.ALLOWED_TYPES:
                newname = 'With' + type_.capitalize() + name[1:]
                Mixin = local_globs[type_.capitalize() + 'Mixin']

                class Temp(base, Mixin, unittest.TestCase):
                    pass
                if type_ == 'manager':
                    Temp = hashlib_helper.requires_hashdigest('md5')(Temp)
                Temp.__name__ = Temp.__qualname__ = newname
                Temp.__module__ = __module__
                remote_globs[newname] = Temp
        elif issubclass(base, unittest.TestCase):

            class Temp(base, object):
                pass
            Temp.__name__ = Temp.__qualname__ = name
            Temp.__module__ = __module__
            remote_globs[name] = Temp
    dangling = [None, None]
    old_start_method = [None]

    def setUpModule():
        if False:
            for i in range(10):
                print('nop')
        multiprocessing.set_forkserver_preload(PRELOAD)
        multiprocessing.process._cleanup()
        dangling[0] = multiprocessing.process._dangling.copy()
        dangling[1] = threading._dangling.copy()
        old_start_method[0] = multiprocessing.get_start_method(allow_none=True)
        try:
            multiprocessing.set_start_method(start_method, force=True)
        except ValueError:
            raise unittest.SkipTest(start_method + ' start method not supported')
        if sys.platform.startswith('linux'):
            try:
                lock = multiprocessing.RLock()
            except OSError:
                raise unittest.SkipTest('OSError raises on RLock creation, see issue 3111!')
        check_enough_semaphores()
        util.get_temp_dir()
        multiprocessing.get_logger().setLevel(LOG_LEVEL)

    def tearDownModule():
        if False:
            while True:
                i = 10
        need_sleep = False
        test.support.gc_collect()
        multiprocessing.set_start_method(old_start_method[0], force=True)
        processes = set(multiprocessing.process._dangling) - set(dangling[0])
        if processes:
            need_sleep = True
            test.support.environment_altered = True
            support.print_warning(f'Dangling processes: {processes}')
        processes = None
        threads = set(threading._dangling) - set(dangling[1])
        if threads:
            need_sleep = True
            test.support.environment_altered = True
            support.print_warning(f'Dangling threads: {threads}')
        threads = None
        if need_sleep:
            time.sleep(0.5)
        multiprocessing.util._cleanup_tests()
    remote_globs['setUpModule'] = setUpModule
    remote_globs['tearDownModule'] = tearDownModule