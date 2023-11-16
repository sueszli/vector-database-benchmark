"""
Various tests for synchronization primitives.
"""
import sys
import time
try:
    from thread import start_new_thread, get_ident
except ImportError:
    from _thread import start_new_thread, get_ident
import threading
import unittest
from gevent.testing import support
from gevent.testing.testcase import TimeAssertMixin

def _wait():
    if False:
        return 10
    time.sleep(0.01)

class Bunch(object):
    """
    A bunch of threads.
    """

    def __init__(self, f, n, wait_before_exit=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a bunch of `n` threads running the same function `f`.\n        If `wait_before_exit` is True, the threads won't terminate until\n        do_finish() is called.\n        "
        self.f = f
        self.n = n
        self.started = []
        self.finished = []
        self._can_exit = not wait_before_exit

        def task():
            if False:
                return 10
            tid = get_ident()
            self.started.append(tid)
            try:
                f()
            finally:
                self.finished.append(tid)
                while not self._can_exit:
                    _wait()
        for _ in range(n):
            start_new_thread(task, ())

    def wait_for_started(self):
        if False:
            print('Hello World!')
        while len(self.started) < self.n:
            _wait()

    def wait_for_finished(self):
        if False:
            i = 10
            return i + 15
        while len(self.finished) < self.n:
            _wait()

    def do_finish(self):
        if False:
            i = 10
            return i + 15
        self._can_exit = True

class BaseTestCase(TimeAssertMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._threads = support.threading_setup()

    def tearDown(self):
        if False:
            while True:
                i = 10
        support.threading_cleanup(*self._threads)
        support.reap_children()

class BaseLockTests(BaseTestCase):
    """
    Tests for both recursive and non-recursive locks.
    """

    def locktype(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def test_constructor(self):
        if False:
            i = 10
            return i + 15
        lock = self.locktype()
        del lock

    def test_acquire_destroy(self):
        if False:
            print('Hello World!')
        lock = self.locktype()
        lock.acquire()
        del lock

    def test_acquire_release(self):
        if False:
            while True:
                i = 10
        lock = self.locktype()
        lock.acquire()
        lock.release()
        del lock

    def test_try_acquire(self):
        if False:
            print('Hello World!')
        lock = self.locktype()
        self.assertTrue(lock.acquire(False))
        lock.release()

    def test_try_acquire_contended(self):
        if False:
            return 10
        lock = self.locktype()
        lock.acquire()
        result = []

        def f():
            if False:
                return 10
            result.append(lock.acquire(False))
        Bunch(f, 1).wait_for_finished()
        self.assertFalse(result[0])
        lock.release()

    def test_acquire_contended(self):
        if False:
            while True:
                i = 10
        lock = self.locktype()
        lock.acquire()
        N = 5

        def f():
            if False:
                print('Hello World!')
            lock.acquire()
            lock.release()
        b = Bunch(f, N)
        b.wait_for_started()
        _wait()
        self.assertEqual(len(b.finished), 0)
        lock.release()
        b.wait_for_finished()
        self.assertEqual(len(b.finished), N)

    def test_with(self):
        if False:
            return 10
        lock = self.locktype()

        def f():
            if False:
                while True:
                    i = 10
            lock.acquire()
            lock.release()

        def _with(err=None):
            if False:
                i = 10
                return i + 15
            with lock:
                if err is not None:
                    raise err
        _with()
        Bunch(f, 1).wait_for_finished()
        self.assertRaises(TypeError, _with, TypeError)
        Bunch(f, 1).wait_for_finished()

    def test_thread_leak(self):
        if False:
            i = 10
            return i + 15
        lock = self.locktype()

        def f():
            if False:
                print('Hello World!')
            lock.acquire()
            lock.release()
        n = len(threading.enumerate())
        Bunch(f, 15).wait_for_finished()
        self.assertEqual(n, len(threading.enumerate()))

class LockTests(BaseLockTests):
    """
    Tests for non-recursive, weak locks
    (which can be acquired and released from different threads).
    """

    def test_reacquire(self):
        if False:
            i = 10
            return i + 15
        lock = self.locktype()
        phase = []

        def f():
            if False:
                print('Hello World!')
            lock.acquire()
            phase.append(None)
            lock.acquire()
            phase.append(None)
        start_new_thread(f, ())
        while not phase:
            _wait()
        _wait()
        self.assertEqual(len(phase), 1)
        lock.release()
        while len(phase) == 1:
            _wait()
        self.assertEqual(len(phase), 2)

    def test_different_thread(self):
        if False:
            while True:
                i = 10
        lock = self.locktype()
        lock.acquire()

        def f():
            if False:
                i = 10
                return i + 15
            lock.release()
        b = Bunch(f, 1)
        b.wait_for_finished()
        lock.acquire()
        lock.release()

class RLockTests(BaseLockTests):
    """
    Tests for recursive locks.
    """

    def test_reacquire(self):
        if False:
            i = 10
            return i + 15
        lock = self.locktype()
        lock.acquire()
        lock.acquire()
        lock.release()
        lock.acquire()
        lock.release()
        lock.release()

    def test_release_unacquired(self):
        if False:
            return 10
        lock = self.locktype()
        self.assertRaises(RuntimeError, lock.release)
        lock.acquire()
        lock.acquire()
        lock.release()
        lock.acquire()
        lock.release()
        lock.release()
        self.assertRaises(RuntimeError, lock.release)

    def test_different_thread(self):
        if False:
            for i in range(10):
                print('nop')
        lock = self.locktype()

        def f():
            if False:
                print('Hello World!')
            lock.acquire()
        b = Bunch(f, 1, True)
        try:
            self.assertRaises(RuntimeError, lock.release)
        finally:
            b.do_finish()

    def test__is_owned(self):
        if False:
            return 10
        lock = self.locktype()
        self.assertFalse(lock._is_owned())
        lock.acquire()
        self.assertTrue(lock._is_owned())
        lock.acquire()
        self.assertTrue(lock._is_owned())
        result = []

        def f():
            if False:
                print('Hello World!')
            result.append(lock._is_owned())
        Bunch(f, 1).wait_for_finished()
        self.assertFalse(result[0])
        lock.release()
        self.assertTrue(lock._is_owned())
        lock.release()
        self.assertFalse(lock._is_owned())

class EventTests(BaseTestCase):
    """
    Tests for Event objects.
    """

    def eventtype(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def test_is_set(self):
        if False:
            print('Hello World!')
        evt = self.eventtype()
        self.assertFalse(evt.is_set())
        evt.set()
        self.assertTrue(evt.is_set())
        evt.set()
        self.assertTrue(evt.is_set())
        evt.clear()
        self.assertFalse(evt.is_set())
        evt.clear()
        self.assertFalse(evt.is_set())

    def _check_notify(self, evt):
        if False:
            while True:
                i = 10
        N = 5
        results1 = []
        results2 = []

        def f():
            if False:
                return 10
            evt.wait()
            results1.append(evt.is_set())
            evt.wait()
            results2.append(evt.is_set())
        b = Bunch(f, N)
        b.wait_for_started()
        _wait()
        self.assertEqual(len(results1), 0)
        evt.set()
        b.wait_for_finished()
        self.assertEqual(results1, [True] * N)
        self.assertEqual(results2, [True] * N)

    def test_notify(self):
        if False:
            for i in range(10):
                print('nop')
        evt = self.eventtype()
        self._check_notify(evt)
        evt.set()
        evt.clear()
        self._check_notify(evt)

    def test_timeout(self):
        if False:
            print('Hello World!')
        evt = self.eventtype()
        results1 = []
        results2 = []
        N = 5

        def f():
            if False:
                for i in range(10):
                    print('nop')
            evt.wait(0.0)
            results1.append(evt.is_set())
            t1 = time.time()
            evt.wait(0.2)
            r = evt.is_set()
            t2 = time.time()
            results2.append((r, t2 - t1))
        Bunch(f, N).wait_for_finished()
        self.assertEqual(results1, [False] * N)
        for (r, dt) in results2:
            self.assertFalse(r)
            self.assertTimeWithinRange(dt, 0.18, 10)
        results1 = []
        results2 = []
        evt.set()
        Bunch(f, N).wait_for_finished()
        self.assertEqual(results1, [True] * N)
        for (r, dt) in results2:
            self.assertTrue(r)

class ConditionTests(BaseTestCase):
    """
    Tests for condition variables.
    """

    def condtype(self, *args):
        if False:
            return 10
        raise NotImplementedError()

    def test_acquire(self):
        if False:
            for i in range(10):
                print('nop')
        cond = self.condtype()
        cond.acquire()
        cond.acquire()
        cond.release()
        cond.release()
        lock = threading.Lock()
        cond = self.condtype(lock)
        cond.acquire()
        self.assertFalse(lock.acquire(False))
        cond.release()
        self.assertTrue(lock.acquire(False))
        self.assertFalse(cond.acquire(False))
        lock.release()
        with cond:
            self.assertFalse(lock.acquire(False))

    def test_unacquired_wait(self):
        if False:
            while True:
                i = 10
        cond = self.condtype()
        self.assertRaises(RuntimeError, cond.wait)

    def test_unacquired_notify(self):
        if False:
            i = 10
            return i + 15
        cond = self.condtype()
        self.assertRaises(RuntimeError, cond.notify)

    def _check_notify(self, cond):
        if False:
            for i in range(10):
                print('nop')
        N = 5
        results1 = []
        results2 = []
        phase_num = 0

        def f():
            if False:
                print('Hello World!')
            cond.acquire()
            cond.wait()
            cond.release()
            results1.append(phase_num)
            cond.acquire()
            cond.wait()
            cond.release()
            results2.append(phase_num)
        b = Bunch(f, N)
        b.wait_for_started()
        _wait()
        self.assertEqual(results1, [])
        cond.acquire()
        cond.notify(3)
        _wait()
        phase_num = 1
        cond.release()
        while len(results1) < 3:
            _wait()
        self.assertEqual(results1, [1] * 3)
        self.assertEqual(results2, [])
        cond.acquire()
        cond.notify(5)
        _wait()
        phase_num = 2
        cond.release()
        while len(results1) + len(results2) < 8:
            _wait()
        self.assertEqual(results1, [1] * 3 + [2] * 2)
        self.assertEqual(results2, [2] * 3)
        cond.acquire()
        cond.notify_all()
        _wait()
        phase_num = 3
        cond.release()
        while len(results2) < 5:
            _wait()
        self.assertEqual(results1, [1] * 3 + [2] * 2)
        self.assertEqual(results2, [2] * 3 + [3] * 2)
        b.wait_for_finished()

    def test_notify(self):
        if False:
            i = 10
            return i + 15
        cond = self.condtype()
        self._check_notify(cond)
        self._check_notify(cond)

    def test_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        cond = self.condtype()
        results = []
        N = 5

        def f():
            if False:
                i = 10
                return i + 15
            cond.acquire()
            t1 = time.time()
            cond.wait(0.2)
            t2 = time.time()
            cond.release()
            results.append(t2 - t1)
        Bunch(f, N).wait_for_finished()
        self.assertEqual(len(results), 5)
        for dt in results:
            self.assertTimeWithinRange(dt, 0.19, 2.0)

class BaseSemaphoreTests(BaseTestCase):
    """
    Common tests for {bounded, unbounded} semaphore objects.
    """

    def semtype(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def test_constructor(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, self.semtype, value=-1)
        self.assertRaises((ValueError, OverflowError), self.semtype, value=-getattr(sys, 'maxint', getattr(sys, 'maxsize', None)))

    def test_acquire(self):
        if False:
            print('Hello World!')
        sem = self.semtype(1)
        sem.acquire()
        sem.release()
        sem = self.semtype(2)
        sem.acquire()
        sem.acquire()
        sem.release()
        sem.release()

    def test_acquire_destroy(self):
        if False:
            i = 10
            return i + 15
        sem = self.semtype()
        sem.acquire()
        del sem

    def test_acquire_contended(self):
        if False:
            i = 10
            return i + 15
        sem = self.semtype(7)
        sem.acquire()
        results1 = []
        results2 = []
        phase_num = 0

        def f():
            if False:
                for i in range(10):
                    print('nop')
            sem.acquire()
            results1.append(phase_num)
            sem.acquire()
            results2.append(phase_num)
        b = Bunch(f, 10)
        b.wait_for_started()
        while len(results1) + len(results2) < 6:
            _wait()
        self.assertEqual(results1 + results2, [0] * 6)
        phase_num = 1
        for _ in range(7):
            sem.release()
        while len(results1) + len(results2) < 13:
            _wait()
        self.assertEqual(sorted(results1 + results2), [0] * 6 + [1] * 7)
        phase_num = 2
        for _ in range(6):
            sem.release()
        while len(results1) + len(results2) < 19:
            _wait()
        self.assertEqual(sorted(results1 + results2), [0] * 6 + [1] * 7 + [2] * 6)
        self.assertFalse(sem.acquire(False))
        sem.release()
        b.wait_for_finished()

    def test_try_acquire(self):
        if False:
            while True:
                i = 10
        sem = self.semtype(2)
        self.assertTrue(sem.acquire(False))
        self.assertTrue(sem.acquire(False))
        self.assertFalse(sem.acquire(False))
        sem.release()
        self.assertTrue(sem.acquire(False))

    def test_try_acquire_contended(self):
        if False:
            i = 10
            return i + 15
        sem = self.semtype(4)
        sem.acquire()
        results = []

        def f():
            if False:
                for i in range(10):
                    print('nop')
            results.append(sem.acquire(False))
            results.append(sem.acquire(False))
        Bunch(f, 5).wait_for_finished()
        self.assertEqual(sorted(results), [False] * 7 + [True] * 3)

    def test_default_value(self):
        if False:
            print('Hello World!')
        sem = self.semtype()
        sem.acquire()

        def f():
            if False:
                while True:
                    i = 10
            sem.acquire()
            sem.release()
        b = Bunch(f, 1)
        b.wait_for_started()
        _wait()
        self.assertFalse(b.finished)
        sem.release()
        b.wait_for_finished()

    def test_with(self):
        if False:
            while True:
                i = 10
        sem = self.semtype(2)

        def _with(err=None):
            if False:
                i = 10
                return i + 15
            with sem:
                self.assertTrue(sem.acquire(False))
                sem.release()
                with sem:
                    self.assertFalse(sem.acquire(False))
                    if err:
                        raise err
        _with()
        self.assertTrue(sem.acquire(False))
        sem.release()
        self.assertRaises(TypeError, _with, TypeError)
        self.assertTrue(sem.acquire(False))
        sem.release()

class SemaphoreTests(BaseSemaphoreTests):
    """
    Tests for unbounded semaphores.
    """

    def test_release_unacquired(self):
        if False:
            print('Hello World!')
        sem = self.semtype(1)
        sem.release()
        sem.acquire()
        sem.acquire()
        sem.release()

class BoundedSemaphoreTests(BaseSemaphoreTests):
    """
    Tests for bounded semaphores.
    """

    def test_release_unacquired(self):
        if False:
            print('Hello World!')
        sem = self.semtype()
        self.assertRaises(ValueError, sem.release)
        sem.acquire()
        sem.release()
        self.assertRaises(ValueError, sem.release)

class BarrierTests(BaseTestCase):
    """
    Tests for Barrier objects.
    """
    N = 5
    defaultTimeout = 2.0

    def setUp(self):
        if False:
            while True:
                i = 10
        self.barrier = self.barriertype(self.N, timeout=self.defaultTimeout)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.barrier.abort()

    def run_threads(self, f):
        if False:
            i = 10
            return i + 15
        b = Bunch(f, self.N - 1)
        f()
        b.wait_for_finished()

    def multipass(self, results, n):
        if False:
            return 10
        m = self.barrier.parties
        self.assertEqual(m, self.N)
        for i in range(n):
            results[0].append(True)
            self.assertEqual(len(results[1]), i * m)
            self.barrier.wait()
            results[1].append(True)
            self.assertEqual(len(results[0]), (i + 1) * m)
            self.barrier.wait()
        self.assertEqual(self.barrier.n_waiting, 0)
        self.assertFalse(self.barrier.broken)

    def test_barrier(self, passes=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a barrier is passed in lockstep\n        '
        results = [[], []]

        def f():
            if False:
                while True:
                    i = 10
            self.multipass(results, passes)
        self.run_threads(f)

    def test_barrier_10(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a barrier works for 10 consecutive runs\n        '
        return self.test_barrier(10)

    def test_wait_return(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test the return value from barrier.wait\n        '
        results = []

        def f():
            if False:
                i = 10
                return i + 15
            r = self.barrier.wait()
            results.append(r)
        self.run_threads(f)
        self.assertEqual(sum(results), sum(range(self.N)))

    def test_action(self):
        if False:
            i = 10
            return i + 15
        "\n        Test the 'action' callback\n        "
        results = []

        def action():
            if False:
                print('Hello World!')
            results.append(True)
        barrier = self.barriertype(self.N, action)

        def f():
            if False:
                while True:
                    i = 10
            barrier.wait()
            self.assertEqual(len(results), 1)
        self.run_threads(f)

    def test_abort(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that an abort will put the barrier in a broken state\n        '
        results1 = []
        results2 = []

        def f():
            if False:
                while True:
                    i = 10
            try:
                i = self.barrier.wait()
                if i == self.N // 2:
                    raise RuntimeError
                self.barrier.wait()
                results1.append(True)
            except threading.BrokenBarrierError:
                results2.append(True)
            except RuntimeError:
                self.barrier.abort()
        self.run_threads(f)
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertTrue(self.barrier.broken)

    def test_reset(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that a 'reset' on a barrier frees the waiting threads\n        "
        results1 = []
        results2 = []
        results3 = []

        def f():
            if False:
                while True:
                    i = 10
            i = self.barrier.wait()
            if i == self.N // 2:
                while self.barrier.n_waiting < self.N - 1:
                    time.sleep(0.001)
                self.barrier.reset()
            else:
                try:
                    self.barrier.wait()
                    results1.append(True)
                except threading.BrokenBarrierError:
                    results2.append(True)
            self.barrier.wait()
            results3.append(True)
        self.run_threads(f)
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertEqual(len(results3), self.N)

    def test_abort_and_reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a barrier can be reset after being broken.\n        '
        results1 = []
        results2 = []
        results3 = []
        barrier2 = self.barriertype(self.N)

        def f():
            if False:
                return 10
            try:
                i = self.barrier.wait()
                if i == self.N // 2:
                    raise RuntimeError
                self.barrier.wait()
                results1.append(True)
            except threading.BrokenBarrierError:
                results2.append(True)
            except RuntimeError:
                self.barrier.abort()
            if barrier2.wait() == self.N // 2:
                self.barrier.reset()
            barrier2.wait()
            self.barrier.wait()
            results3.append(True)
        self.run_threads(f)
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N - 1)
        self.assertEqual(len(results3), self.N)

    def test_timeout(self):
        if False:
            i = 10
            return i + 15
        '\n        Test wait(timeout)\n        '

        def f():
            if False:
                for i in range(10):
                    print('nop')
            i = self.barrier.wait()
            if i == self.N // 2:
                time.sleep(1.0)
            self.assertRaises(threading.BrokenBarrierError, self.barrier.wait, 0.5)
        self.run_threads(f)

    def test_default_timeout(self):
        if False:
            return 10
        "\n        Test the barrier's default timeout\n        "
        barrier = self.barriertype(self.N, timeout=0.3)

        def f():
            if False:
                while True:
                    i = 10
            i = barrier.wait()
            if i == self.N // 2:
                time.sleep(1.0)
            self.assertRaises(threading.BrokenBarrierError, barrier.wait)
        self.run_threads(f)

    def test_single_thread(self):
        if False:
            i = 10
            return i + 15
        b = self.barriertype(1)
        b.wait()
        b.wait()
if __name__ == '__main__':
    print('This module contains no tests; it is used by other test cases like test_threading_2')