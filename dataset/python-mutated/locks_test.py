import asyncio
from datetime import timedelta
import typing
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase

class ConditionTest(AsyncTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.history = []

    def record_done(self, future, key):
        if False:
            for i in range(10):
                print('nop')
        'Record the resolution of a Future returned by Condition.wait.'

        def callback(_):
            if False:
                for i in range(10):
                    print('nop')
            if not future.result():
                self.history.append('timeout')
            else:
                self.history.append(key)
        future.add_done_callback(callback)

    def loop_briefly(self):
        if False:
            for i in range(10):
                print('nop')
        'Run all queued callbacks on the IOLoop.\n\n        In these tests, this method is used after calling notify() to\n        preserve the pre-5.0 behavior in which callbacks ran\n        synchronously.\n        '
        self.io_loop.add_callback(self.stop)
        self.wait()

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        c = locks.Condition()
        self.assertIn('Condition', repr(c))
        self.assertNotIn('waiters', repr(c))
        c.wait()
        self.assertIn('waiters', repr(c))

    @gen_test
    def test_notify(self):
        if False:
            i = 10
            return i + 15
        c = locks.Condition()
        self.io_loop.call_later(0.01, c.notify)
        yield c.wait()

    def test_notify_1(self):
        if False:
            i = 10
            return i + 15
        c = locks.Condition()
        self.record_done(c.wait(), 'wait1')
        self.record_done(c.wait(), 'wait2')
        c.notify(1)
        self.loop_briefly()
        self.history.append('notify1')
        c.notify(1)
        self.loop_briefly()
        self.history.append('notify2')
        self.assertEqual(['wait1', 'notify1', 'wait2', 'notify2'], self.history)

    def test_notify_n(self):
        if False:
            print('Hello World!')
        c = locks.Condition()
        for i in range(6):
            self.record_done(c.wait(), i)
        c.notify(3)
        self.loop_briefly()
        self.assertEqual(list(range(3)), self.history)
        c.notify(1)
        self.loop_briefly()
        self.assertEqual(list(range(4)), self.history)
        c.notify(2)
        self.loop_briefly()
        self.assertEqual(list(range(6)), self.history)

    def test_notify_all(self):
        if False:
            print('Hello World!')
        c = locks.Condition()
        for i in range(4):
            self.record_done(c.wait(), i)
        c.notify_all()
        self.loop_briefly()
        self.history.append('notify_all')
        self.assertEqual(list(range(4)) + ['notify_all'], self.history)

    @gen_test
    def test_wait_timeout(self):
        if False:
            return 10
        c = locks.Condition()
        wait = c.wait(timedelta(seconds=0.01))
        self.io_loop.call_later(0.02, c.notify)
        yield gen.sleep(0.03)
        self.assertFalse((yield wait))

    @gen_test
    def test_wait_timeout_preempted(self):
        if False:
            while True:
                i = 10
        c = locks.Condition()
        self.io_loop.call_later(0.01, c.notify)
        wait = c.wait(timedelta(seconds=0.02))
        yield gen.sleep(0.03)
        yield wait

    @gen_test
    def test_notify_n_with_timeout(self):
        if False:
            return 10
        c = locks.Condition()
        self.record_done(c.wait(), 0)
        self.record_done(c.wait(timedelta(seconds=0.01)), 1)
        self.record_done(c.wait(), 2)
        self.record_done(c.wait(), 3)
        yield gen.sleep(0.02)
        self.assertEqual(['timeout'], self.history)
        c.notify(2)
        yield gen.sleep(0.01)
        self.assertEqual(['timeout', 0, 2], self.history)
        self.assertEqual(['timeout', 0, 2], self.history)
        c.notify()
        yield
        self.assertEqual(['timeout', 0, 2, 3], self.history)

    @gen_test
    def test_notify_all_with_timeout(self):
        if False:
            i = 10
            return i + 15
        c = locks.Condition()
        self.record_done(c.wait(), 0)
        self.record_done(c.wait(timedelta(seconds=0.01)), 1)
        self.record_done(c.wait(), 2)
        yield gen.sleep(0.02)
        self.assertEqual(['timeout'], self.history)
        c.notify_all()
        yield
        self.assertEqual(['timeout', 0, 2], self.history)

    @gen_test
    def test_nested_notify(self):
        if False:
            return 10
        c = locks.Condition()
        futures = [asyncio.ensure_future(c.wait()) for _ in range(3)]
        futures[1].add_done_callback(lambda _: c.notify())
        c.notify(2)
        yield
        self.assertTrue(all((f.done() for f in futures)))

    @gen_test
    def test_garbage_collection(self):
        if False:
            for i in range(10):
                print('nop')
        c = locks.Condition()
        for _ in range(101):
            c.wait(timedelta(seconds=0.01))
        future = asyncio.ensure_future(c.wait())
        self.assertEqual(102, len(c._waiters))
        yield gen.sleep(0.02)
        self.assertEqual(1, len(c._waiters))
        self.assertFalse(future.done())
        c.notify()
        self.assertTrue(future.done())

class EventTest(AsyncTestCase):

    def test_repr(self):
        if False:
            return 10
        event = locks.Event()
        self.assertTrue('clear' in str(event))
        self.assertFalse('set' in str(event))
        event.set()
        self.assertFalse('clear' in str(event))
        self.assertTrue('set' in str(event))

    def test_event(self):
        if False:
            print('Hello World!')
        e = locks.Event()
        future_0 = asyncio.ensure_future(e.wait())
        e.set()
        future_1 = asyncio.ensure_future(e.wait())
        e.clear()
        future_2 = asyncio.ensure_future(e.wait())
        self.assertTrue(future_0.done())
        self.assertTrue(future_1.done())
        self.assertFalse(future_2.done())

    @gen_test
    def test_event_timeout(self):
        if False:
            print('Hello World!')
        e = locks.Event()
        with self.assertRaises(TimeoutError):
            yield e.wait(timedelta(seconds=0.01))
        self.io_loop.add_timeout(timedelta(seconds=0.01), e.set)
        yield e.wait(timedelta(seconds=1))

    def test_event_set_multiple(self):
        if False:
            i = 10
            return i + 15
        e = locks.Event()
        e.set()
        e.set()
        self.assertTrue(e.is_set())

    def test_event_wait_clear(self):
        if False:
            i = 10
            return i + 15
        e = locks.Event()
        f0 = asyncio.ensure_future(e.wait())
        e.clear()
        f1 = asyncio.ensure_future(e.wait())
        e.set()
        self.assertTrue(f0.done())
        self.assertTrue(f1.done())

class SemaphoreTest(AsyncTestCase):

    def test_negative_value(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, locks.Semaphore, value=-1)

    def test_repr(self):
        if False:
            print('Hello World!')
        sem = locks.Semaphore()
        self.assertIn('Semaphore', repr(sem))
        self.assertIn('unlocked,value:1', repr(sem))
        sem.acquire()
        self.assertIn('locked', repr(sem))
        self.assertNotIn('waiters', repr(sem))
        sem.acquire()
        self.assertIn('waiters', repr(sem))

    def test_acquire(self):
        if False:
            i = 10
            return i + 15
        sem = locks.Semaphore()
        f0 = asyncio.ensure_future(sem.acquire())
        self.assertTrue(f0.done())
        f1 = asyncio.ensure_future(sem.acquire())
        self.assertFalse(f1.done())
        f2 = asyncio.ensure_future(sem.acquire())
        sem.release()
        self.assertTrue(f1.done())
        self.assertFalse(f2.done())
        sem.release()
        self.assertTrue(f2.done())
        sem.release()
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertEqual(0, len(sem._waiters))

    @gen_test
    def test_acquire_timeout(self):
        if False:
            return 10
        sem = locks.Semaphore(2)
        yield sem.acquire()
        yield sem.acquire()
        acquire = sem.acquire(timedelta(seconds=0.01))
        self.io_loop.call_later(0.02, sem.release)
        yield gen.sleep(0.3)
        with self.assertRaises(gen.TimeoutError):
            yield acquire
        sem.acquire()
        f = asyncio.ensure_future(sem.acquire())
        self.assertFalse(f.done())
        sem.release()
        self.assertTrue(f.done())

    @gen_test
    def test_acquire_timeout_preempted(self):
        if False:
            return 10
        sem = locks.Semaphore(1)
        yield sem.acquire()
        self.io_loop.call_later(0.01, sem.release)
        acquire = sem.acquire(timedelta(seconds=0.02))
        yield gen.sleep(0.03)
        yield acquire

    def test_release_unacquired(self):
        if False:
            return 10
        sem = locks.Semaphore()
        sem.release()
        sem.release()
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertFalse(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_garbage_collection(self):
        if False:
            return 10
        sem = locks.Semaphore(value=0)
        futures = [asyncio.ensure_future(sem.acquire(timedelta(seconds=0.01))) for _ in range(101)]
        future = asyncio.ensure_future(sem.acquire())
        self.assertEqual(102, len(sem._waiters))
        yield gen.sleep(0.02)
        self.assertEqual(1, len(sem._waiters))
        self.assertFalse(future.done())
        sem.release()
        self.assertTrue(future.done())
        for future in futures:
            self.assertRaises(TimeoutError, future.result)

class SemaphoreContextManagerTest(AsyncTestCase):

    @gen_test
    def test_context_manager(self):
        if False:
            while True:
                i = 10
        sem = locks.Semaphore()
        with (yield sem.acquire()) as yielded:
            self.assertTrue(yielded is None)
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_context_manager_async_await(self):
        if False:
            print('Hello World!')
        sem = locks.Semaphore()

        async def f():
            async with sem as yielded:
                self.assertTrue(yielded is None)
        yield f()
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_context_manager_exception(self):
        if False:
            return 10
        sem = locks.Semaphore()
        with self.assertRaises(ZeroDivisionError):
            with (yield sem.acquire()):
                1 / 0
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_context_manager_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        sem = locks.Semaphore()
        with (yield sem.acquire(timedelta(seconds=0.01))):
            pass
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_context_manager_timeout_error(self):
        if False:
            return 10
        sem = locks.Semaphore(value=0)
        with self.assertRaises(gen.TimeoutError):
            with (yield sem.acquire(timedelta(seconds=0.01))):
                pass
        self.assertFalse(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_context_manager_contended(self):
        if False:
            while True:
                i = 10
        sem = locks.Semaphore()
        history = []

        @gen.coroutine
        def f(index):
            if False:
                print('Hello World!')
            with (yield sem.acquire()):
                history.append('acquired %d' % index)
                yield gen.sleep(0.01)
                history.append('release %d' % index)
        yield [f(i) for i in range(2)]
        expected_history = []
        for i in range(2):
            expected_history.extend(['acquired %d' % i, 'release %d' % i])
        self.assertEqual(expected_history, history)

    @gen_test
    def test_yield_sem(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(gen.BadYieldError):
            with (yield locks.Semaphore()):
                pass

    def test_context_manager_misuse(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(RuntimeError):
            with locks.Semaphore():
                pass

class BoundedSemaphoreTest(AsyncTestCase):

    def test_release_unacquired(self):
        if False:
            print('Hello World!')
        sem = locks.BoundedSemaphore()
        self.assertRaises(ValueError, sem.release)
        sem.acquire()
        future = asyncio.ensure_future(sem.acquire())
        self.assertFalse(future.done())
        sem.release()
        self.assertTrue(future.done())
        sem.release()
        self.assertRaises(ValueError, sem.release)

class LockTests(AsyncTestCase):

    def test_repr(self):
        if False:
            print('Hello World!')
        lock = locks.Lock()
        repr(lock)
        lock.acquire()
        repr(lock)

    def test_acquire_release(self):
        if False:
            return 10
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        future = asyncio.ensure_future(lock.acquire())
        self.assertFalse(future.done())
        lock.release()
        self.assertTrue(future.done())

    @gen_test
    def test_acquire_fifo(self):
        if False:
            while True:
                i = 10
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        N = 5
        history = []

        @gen.coroutine
        def f(idx):
            if False:
                while True:
                    i = 10
            with (yield lock.acquire()):
                history.append(idx)
        futures = [f(i) for i in range(N)]
        self.assertFalse(any((future.done() for future in futures)))
        lock.release()
        yield futures
        self.assertEqual(list(range(N)), history)

    @gen_test
    def test_acquire_fifo_async_with(self):
        if False:
            for i in range(10):
                print('nop')
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        N = 5
        history = []

        async def f(idx):
            async with lock:
                history.append(idx)
        futures = [f(i) for i in range(N)]
        lock.release()
        yield futures
        self.assertEqual(list(range(N)), history)

    @gen_test
    def test_acquire_timeout(self):
        if False:
            return 10
        lock = locks.Lock()
        lock.acquire()
        with self.assertRaises(gen.TimeoutError):
            yield lock.acquire(timeout=timedelta(seconds=0.01))
        self.assertFalse(asyncio.ensure_future(lock.acquire()).done())

    def test_multi_release(self):
        if False:
            for i in range(10):
                print('nop')
        lock = locks.Lock()
        self.assertRaises(RuntimeError, lock.release)
        lock.acquire()
        lock.release()
        self.assertRaises(RuntimeError, lock.release)

    @gen_test
    def test_yield_lock(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(gen.BadYieldError):
            with (yield locks.Lock()):
                pass

    def test_context_manager_misuse(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(RuntimeError):
            with locks.Lock():
                pass
if __name__ == '__main__':
    unittest.main()