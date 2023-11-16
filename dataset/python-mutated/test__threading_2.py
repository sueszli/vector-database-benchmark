from __future__ import print_function
from gevent.testing.six import xrange
import gevent.testing as greentest
setup_ = "from gevent import monkey; monkey.patch_all()\nfrom gevent.event import Event\nfrom gevent.lock import RLock, Semaphore, BoundedSemaphore\nfrom gevent.thread import allocate_lock as Lock\nimport threading\nthreading.Event = Event\nthreading.Lock = Lock\n# NOTE: We're completely patching around the allocate_lock\n# patch we try to do with RLock; our monkey patch doesn't\n# behave this way, but we do it in tests to make sure that\n# our RLock implementation behaves correctly by itself.\n# However, we must test the patched version too, so make it\n# available.\nthreading.NativeRLock = threading.RLock\nthreading.RLock = RLock\nthreading.Semaphore = Semaphore\nthreading.BoundedSemaphore = BoundedSemaphore\n"
exec(setup_)
setup_3 = '\n'.join(('            %s' % line for line in setup_.split('\n')))
setup_4 = '\n'.join(('                %s' % line for line in setup_.split('\n')))
from gevent.testing import support
verbose = support.verbose
import random
import re
import sys
import threading
try:
    import thread
except ImportError:
    import _thread as thread
import time
import unittest
import weakref
from gevent.tests import lock_tests
verbose = False

def skipDueToHang(cls):
    if False:
        while True:
            i = 10
    return unittest.skipIf(greentest.PYPY3 and greentest.RUNNING_ON_CI, 'SKIPPED: Timeout on PyPy3 on Travis')(cls)

class Counter(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.value = 0

    def inc(self):
        if False:
            return 10
        self.value += 1

    def dec(self):
        if False:
            i = 10
            return i + 15
        self.value -= 1

    def get(self):
        if False:
            return 10
        return self.value

class TestThread(threading.Thread):

    def __init__(self, name, testcase, sema, mutex, nrunning):
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self, name=name)
        self.testcase = testcase
        self.sema = sema
        self.mutex = mutex
        self.nrunning = nrunning

    def run(self):
        if False:
            while True:
                i = 10
        delay = random.random() / 10000.0
        if verbose:
            print('task %s will run for %.1f usec' % (self.name, delay * 1000000.0))
        with self.sema:
            with self.mutex:
                self.nrunning.inc()
                if verbose:
                    print(self.nrunning.get(), 'tasks are running')
                self.testcase.assertLessEqual(self.nrunning.get(), 3)
            time.sleep(delay)
            if verbose:
                print('task', self.name, 'done')
            with self.mutex:
                self.nrunning.dec()
                self.testcase.assertGreaterEqual(self.nrunning.get(), 0)
                if verbose:
                    print('%s is finished. %d tasks are running' % (self.name, self.nrunning.get()))

@skipDueToHang
class ThreadTests(unittest.TestCase):

    def test_various_ops(self):
        if False:
            for i in range(10):
                print('nop')
        NUMTASKS = 10
        sema = threading.BoundedSemaphore(value=3)
        mutex = threading.RLock()
        numrunning = Counter()
        threads = []
        for i in range(NUMTASKS):
            t = TestThread('<thread %d>' % i, self, sema, mutex, numrunning)
            threads.append(t)
            t.daemon = False
            if hasattr(t, 'ident'):
                self.assertIsNone(t.ident)
                self.assertFalse(t.daemon)
                self.assertTrue(re.match('<TestThread\\(.*, initial\\)>', repr(t)))
            t.start()
        if verbose:
            print('waiting for all tasks to complete')
        for t in threads:
            t.join(NUMTASKS)
            self.assertFalse(t.is_alive(), t.__dict__)
            if hasattr(t, 'ident'):
                self.assertNotEqual(t.ident, 0)
                self.assertFalse(t.ident is None)
                self.assertTrue(re.match('<TestThread\\(.*, \\w+ -?\\d+\\)>', repr(t)))
        if verbose:
            print('all tasks done')
        self.assertEqual(numrunning.get(), 0)

    def test_ident_of_no_threading_threads(self):
        if False:
            while True:
                i = 10
        t = threading.current_thread()
        self.assertFalse(t.ident is None)
        str(t)
        repr(t)

        def f():
            if False:
                for i in range(10):
                    print('nop')
            t = threading.current_thread()
            ident.append(t.ident)
            str(t)
            repr(t)
            done.set()
        done = threading.Event()
        ident = []
        thread.start_new_thread(f, ())
        done.wait()
        self.assertFalse(ident[0] is None)
        del threading._active[ident[0]]

    def test_various_ops_small_stack(self):
        if False:
            print('Hello World!')
        if verbose:
            print('with 256kB thread stack size...')
        try:
            threading.stack_size(262144)
        except thread.error:
            if verbose:
                print('platform does not support changing thread stack size')
            return
        self.test_various_ops()
        threading.stack_size(0)

    def test_various_ops_large_stack(self):
        if False:
            i = 10
            return i + 15
        if verbose:
            print('with 1MB thread stack size...')
        try:
            threading.stack_size(1048576)
        except thread.error:
            if verbose:
                print('platform does not support changing thread stack size')
            return
        self.test_various_ops()
        threading.stack_size(0)

    def test_foreign_thread(self):
        if False:
            print('Hello World!')

        def f(mutex):
            if False:
                i = 10
                return i + 15
            threading.current_thread()
            mutex.release()
        mutex = threading.Lock()
        mutex.acquire()
        tid = thread.start_new_thread(f, (mutex,))
        mutex.acquire()
        self.assertIn(tid, threading._active)
        self.assertIsInstance(threading._active[tid], threading._DummyThread)
        del threading._active[tid]

    def SKIP_test_PyThreadState_SetAsyncExc(self):
        if False:
            return 10
        try:
            import ctypes
        except ImportError:
            if verbose:
                print("test_PyThreadState_SetAsyncExc can't import ctypes")
            return
        set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc

        class AsyncExc(Exception):
            pass
        exception = ctypes.py_object(AsyncExc)
        worker_started = threading.Event()
        worker_saw_exception = threading.Event()

        class Worker(threading.Thread):
            id = None
            finished = False

            def run(self):
                if False:
                    return 10
                self.id = thread.get_ident()
                self.finished = False
                try:
                    while True:
                        worker_started.set()
                        time.sleep(0.1)
                except AsyncExc:
                    self.finished = True
                    worker_saw_exception.set()
        t = Worker()
        t.daemon = True
        t.start()
        if verbose:
            print('    started worker thread')
        if verbose:
            print('    trying nonsensical thread id')
        result = set_async_exc(ctypes.c_long(-1), exception)
        self.assertEqual(result, 0)
        if verbose:
            print('    waiting for worker thread to get started')
        worker_started.wait()
        if verbose:
            print("    verifying worker hasn't exited")
        self.assertFalse(t.finished)
        if verbose:
            print('    attempting to raise asynch exception in worker')
        result = set_async_exc(ctypes.c_long(t.id), exception)
        self.assertEqual(result, 1)
        if verbose:
            print('    waiting for worker to say it caught the exception')
        worker_saw_exception.wait(timeout=10)
        self.assertTrue(t.finished)
        if verbose:
            print('    all OK -- joining worker')
        if t.finished:
            t.join()

    def test_limbo_cleanup(self):
        if False:
            i = 10
            return i + 15

        def fail_new_thread(*_args):
            if False:
                i = 10
                return i + 15
            raise thread.error()
        _start_new_thread = threading._start_new_thread
        threading._start_new_thread = fail_new_thread
        try:
            t = threading.Thread(target=lambda : None)
            self.assertRaises(thread.error, t.start)
            self.assertFalse(t in threading._limbo, 'Failed to cleanup _limbo map on failure of Thread.start().')
        finally:
            threading._start_new_thread = _start_new_thread

    def test_finalize_runnning_thread(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import ctypes
            getattr(ctypes, 'pythonapi')
            getattr(ctypes.pythonapi, 'PyGILState_Ensure')
        except (ImportError, AttributeError):
            if verbose:
                print("test_finalize_with_runnning_thread can't import ctypes")
            return
        del ctypes
        import subprocess
        rc = subprocess.call([sys.executable, '-W', 'ignore', '-c', 'if 1:\n%s\n            import ctypes, sys, time\n            try:\n                import thread\n            except ImportError:\n                import _thread as thread # Py3\n\n            # This lock is used as a simple event variable.\n            ready = thread.allocate_lock()\n            ready.acquire()\n\n            # Module globals are cleared before __del__ is run\n            # So we save the functions in class dict\n            class C:\n                ensure = ctypes.pythonapi.PyGILState_Ensure\n                release = ctypes.pythonapi.PyGILState_Release\n                def __del__(self):\n                    state = self.ensure()\n                    self.release(state)\n\n            def waitingThread():\n                x = C()\n                ready.release()\n                time.sleep(100)\n\n            thread.start_new_thread(waitingThread, ())\n            ready.acquire()  # Be sure the other thread is waiting.\n            sys.exit(42)\n            ' % setup_3])
        self.assertEqual(rc, 42)

    @greentest.skipOnLibuvOnPyPyOnWin('hangs')
    def test_join_nondaemon_on_shutdown(self):
        if False:
            while True:
                i = 10
        import subprocess
        script = 'if 1:\n%s\n                import threading\n                from time import sleep\n\n                def child():\n                    sleep(0.3)\n                    # As a non-daemon thread we SHOULD wake up and nothing\n                    # should be torn down yet\n                    print("Woke up, sleep function is: %%s.%%s" %% (sleep.__module__, sleep.__name__))\n\n                threading.Thread(target=child).start()\n                raise SystemExit\n        ' % setup_4
        p = subprocess.Popen([sys.executable, '-W', 'ignore', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        stdout = stdout.strip()
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')
        self.assertEqual('Woke up, sleep function is: gevent.hub.sleep', stdout)

    @greentest.skipIf(not hasattr(sys, 'getcheckinterval'), 'Needs sys.getcheckinterval')
    def test_enumerate_after_join(self):
        if False:
            print('Hello World!')
        enum = threading.enumerate
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            old_interval = sys.getcheckinterval()
            try:
                for i in xrange(1, 100):
                    sys.setcheckinterval(i // 5)
                    t = threading.Thread(target=lambda : None)
                    t.start()
                    t.join()
                    l = enum()
                    self.assertFalse(t in l, '#1703448 triggered after %d trials: %s' % (i, l))
            finally:
                sys.setcheckinterval(old_interval)
    if not hasattr(sys, 'pypy_version_info'):

        def test_no_refcycle_through_target(self):
            if False:
                i = 10
                return i + 15

            class RunSelfFunction(object):

                def __init__(self, should_raise):
                    if False:
                        return 10
                    self.should_raise = should_raise
                    self.thread = threading.Thread(target=self._run, args=(self,), kwargs={'_yet_another': self})
                    self.thread.start()

                def _run(self, _other_ref, _yet_another):
                    if False:
                        print('Hello World!')
                    if self.should_raise:
                        raise SystemExit
            cyclic_object = RunSelfFunction(should_raise=False)
            weak_cyclic_object = weakref.ref(cyclic_object)
            cyclic_object.thread.join()
            del cyclic_object
            self.assertIsNone(weak_cyclic_object(), msg='%d references still around' % sys.getrefcount(weak_cyclic_object()))
            raising_cyclic_object = RunSelfFunction(should_raise=True)
            weak_raising_cyclic_object = weakref.ref(raising_cyclic_object)
            raising_cyclic_object.thread.join()
            del raising_cyclic_object
            self.assertIsNone(weak_raising_cyclic_object(), msg='%d references still around' % sys.getrefcount(weak_raising_cyclic_object()))

@skipDueToHang
class ThreadJoinOnShutdown(unittest.TestCase):

    def _run_and_join(self, script):
        if False:
            for i in range(10):
                print('nop')
        script = "if 1:\n%s\n            import sys, os, time, threading\n            # a thread, which waits for the main program to terminate\n            def joiningfunc(mainthread):\n                mainthread.join()\n                print('end of thread')\n        \n" % setup_3 + script
        import subprocess
        p = subprocess.Popen([sys.executable, '-W', 'ignore', '-c', script], stdout=subprocess.PIPE)
        rc = p.wait()
        data = p.stdout.read().replace(b'\r', b'')
        p.stdout.close()
        self.assertEqual(data, b'end of main\nend of thread\n')
        self.assertNotEqual(rc, 2, b'interpreter was blocked')
        self.assertEqual(rc, 0, b'Unexpected error')

    @greentest.skipOnLibuvOnPyPyOnWin('hangs')
    def test_1_join_on_shutdown(self):
        if False:
            i = 10
            return i + 15
        script = "if 1:\n            import os\n            t = threading.Thread(target=joiningfunc,\n                                 args=(threading.current_thread(),))\n            t.start()\n            time.sleep(0.2)\n            print('end of main')\n            "
        self._run_and_join(script)

    @greentest.skipOnPyPy3OnCI('Sometimes randomly times out')
    def test_2_join_in_forked_process(self):
        if False:
            print('Hello World!')
        import os
        if not hasattr(os, 'fork'):
            return
        script = "if 1:\n            childpid = os.fork()\n            if childpid != 0:\n                os.waitpid(childpid, 0)\n                sys.exit(0)\n\n            t = threading.Thread(target=joiningfunc,\n                                 args=(threading.current_thread(),))\n            t.start()\n            print('end of main')\n            "
        self._run_and_join(script)

    def test_3_join_in_forked_from_thread(self):
        if False:
            return 10
        import os
        if not hasattr(os, 'fork'):
            return
        script = "if 1:\n            main_thread = threading.current_thread()\n            def worker():\n                threading._after_fork = lambda: None\n                childpid = os.fork()\n                if childpid != 0:\n                    os.waitpid(childpid, 0)\n                    sys.exit(0)\n\n                t = threading.Thread(target=joiningfunc,\n                                     args=(main_thread,))\n                print('end of main')\n                t.start()\n                t.join() # Should not block: main_thread is already stopped\n\n            w = threading.Thread(target=worker)\n            w.start()\n            import sys\n            if sys.version_info[:2] >= (3, 7) or (sys.version_info[:2] >= (3, 5) and hasattr(sys, 'pypy_version_info') and sys.platform != 'darwin'):\n                w.join()\n            "
        self._run_and_join(script)

@skipDueToHang
class ThreadingExceptionTests(unittest.TestCase):

    def test_start_thread_again(self):
        if False:
            return 10
        thread_ = threading.Thread()
        thread_.start()
        self.assertRaises(RuntimeError, thread_.start)

    def test_joining_current_thread(self):
        if False:
            return 10
        current_thread = threading.current_thread()
        self.assertRaises(RuntimeError, current_thread.join)

    def test_joining_inactive_thread(self):
        if False:
            i = 10
            return i + 15
        thread_ = threading.Thread()
        self.assertRaises(RuntimeError, thread_.join)

    def test_daemonize_active_thread(self):
        if False:
            while True:
                i = 10
        thread_ = threading.Thread()
        thread_.start()
        self.assertRaises(RuntimeError, setattr, thread_, 'daemon', True)

@skipDueToHang
class LockTests(lock_tests.LockTests):
    locktype = staticmethod(threading.Lock)

@skipDueToHang
class RLockTests(lock_tests.RLockTests):
    locktype = staticmethod(threading.RLock)

@skipDueToHang
class NativeRLockTests(lock_tests.RLockTests):
    locktype = staticmethod(threading.NativeRLock)

@skipDueToHang
class EventTests(lock_tests.EventTests):
    eventtype = staticmethod(threading.Event)

@skipDueToHang
class ConditionAsRLockTests(lock_tests.RLockTests):
    locktype = staticmethod(threading.Condition)

@skipDueToHang
class ConditionTests(lock_tests.ConditionTests):
    condtype = staticmethod(threading.Condition)

@skipDueToHang
class SemaphoreTests(lock_tests.SemaphoreTests):
    semtype = staticmethod(threading.Semaphore)

@skipDueToHang
class BoundedSemaphoreTests(lock_tests.BoundedSemaphoreTests):
    semtype = staticmethod(threading.BoundedSemaphore)
if __name__ == '__main__':
    greentest.main()