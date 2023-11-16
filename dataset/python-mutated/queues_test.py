import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase

class QueueBasicTest(AsyncTestCase):

    def test_repr_and_str(self):
        if False:
            return 10
        q = queues.Queue(maxsize=1)
        self.assertIn(hex(id(q)), repr(q))
        self.assertNotIn(hex(id(q)), str(q))
        q.get()
        for q_str in (repr(q), str(q)):
            self.assertTrue(q_str.startswith('<Queue'))
            self.assertIn('maxsize=1', q_str)
            self.assertIn('getters[1]', q_str)
            self.assertNotIn('putters', q_str)
            self.assertNotIn('tasks', q_str)
        q.put(None)
        q.put(None)
        q.put(None)
        for q_str in (repr(q), str(q)):
            self.assertNotIn('getters', q_str)
            self.assertIn('putters[1]', q_str)
            self.assertIn('tasks=2', q_str)

    def test_order(self):
        if False:
            i = 10
            return i + 15
        q = queues.Queue()
        for i in [1, 3, 2]:
            q.put_nowait(i)
        items = [q.get_nowait() for _ in range(3)]
        self.assertEqual([1, 3, 2], items)

    @gen_test
    def test_maxsize(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, queues.Queue, maxsize=None)
        self.assertRaises(ValueError, queues.Queue, maxsize=-1)
        q = queues.Queue(maxsize=2)
        self.assertTrue(q.empty())
        self.assertFalse(q.full())
        self.assertEqual(2, q.maxsize)
        self.assertTrue(q.put(0).done())
        self.assertTrue(q.put(1).done())
        self.assertFalse(q.empty())
        self.assertTrue(q.full())
        put2 = q.put(2)
        self.assertFalse(put2.done())
        self.assertEqual(0, (yield q.get()))
        self.assertTrue(put2.done())
        self.assertFalse(q.empty())
        self.assertTrue(q.full())

class QueueGetTest(AsyncTestCase):

    @gen_test
    def test_blocking_get(self):
        if False:
            return 10
        q = queues.Queue()
        q.put_nowait(0)
        self.assertEqual(0, (yield q.get()))

    def test_nonblocking_get(self):
        if False:
            print('Hello World!')
        q = queues.Queue()
        q.put_nowait(0)
        self.assertEqual(0, q.get_nowait())

    def test_nonblocking_get_exception(self):
        if False:
            for i in range(10):
                print('nop')
        q = queues.Queue()
        self.assertRaises(queues.QueueEmpty, q.get_nowait)

    @gen_test
    def test_get_with_putters(self):
        if False:
            print('Hello World!')
        q = queues.Queue(1)
        q.put_nowait(0)
        put = q.put(1)
        self.assertEqual(0, (yield q.get()))
        self.assertIsNone((yield put))

    @gen_test
    def test_blocking_get_wait(self):
        if False:
            print('Hello World!')
        q = queues.Queue()
        q.put(0)
        self.io_loop.call_later(0.01, q.put_nowait, 1)
        self.io_loop.call_later(0.02, q.put_nowait, 2)
        self.assertEqual(0, (yield q.get(timeout=timedelta(seconds=1))))
        self.assertEqual(1, (yield q.get(timeout=timedelta(seconds=1))))

    @gen_test
    def test_get_timeout(self):
        if False:
            print('Hello World!')
        q = queues.Queue()
        get_timeout = q.get(timeout=timedelta(seconds=0.01))
        get = q.get()
        with self.assertRaises(TimeoutError):
            yield get_timeout
        q.put_nowait(0)
        self.assertEqual(0, (yield get))

    @gen_test
    def test_get_timeout_preempted(self):
        if False:
            for i in range(10):
                print('nop')
        q = queues.Queue()
        get = q.get(timeout=timedelta(seconds=0.01))
        q.put(0)
        yield gen.sleep(0.02)
        self.assertEqual(0, (yield get))

    @gen_test
    def test_get_clears_timed_out_putters(self):
        if False:
            i = 10
            return i + 15
        q = queues.Queue(1)
        putters = [q.put(i, timedelta(seconds=0.01)) for i in range(10)]
        put = q.put(10)
        self.assertEqual(10, len(q._putters))
        yield gen.sleep(0.02)
        self.assertEqual(10, len(q._putters))
        self.assertFalse(put.done())
        q.put(11)
        self.assertEqual(0, (yield q.get()))
        self.assertEqual(1, len(q._putters))
        for putter in putters[1:]:
            self.assertRaises(TimeoutError, putter.result)

    @gen_test
    def test_get_clears_timed_out_getters(self):
        if False:
            print('Hello World!')
        q = queues.Queue()
        getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
        get = asyncio.ensure_future(q.get())
        self.assertEqual(11, len(q._getters))
        yield gen.sleep(0.02)
        self.assertEqual(11, len(q._getters))
        self.assertFalse(get.done())
        q.get()
        self.assertEqual(2, len(q._getters))
        for getter in getters:
            self.assertRaises(TimeoutError, getter.result)

    @gen_test
    def test_async_for(self):
        if False:
            print('Hello World!')
        q = queues.Queue()
        for i in range(5):
            q.put(i)

        async def f():
            results = []
            async for i in q:
                results.append(i)
                if i == 4:
                    return results
        results = (yield f())
        self.assertEqual(results, list(range(5)))

class QueuePutTest(AsyncTestCase):

    @gen_test
    def test_blocking_put(self):
        if False:
            for i in range(10):
                print('nop')
        q = queues.Queue()
        q.put(0)
        self.assertEqual(0, q.get_nowait())

    def test_nonblocking_put_exception(self):
        if False:
            return 10
        q = queues.Queue(1)
        q.put(0)
        self.assertRaises(queues.QueueFull, q.put_nowait, 1)

    @gen_test
    def test_put_with_getters(self):
        if False:
            i = 10
            return i + 15
        q = queues.Queue()
        get0 = q.get()
        get1 = q.get()
        yield q.put(0)
        self.assertEqual(0, (yield get0))
        yield q.put(1)
        self.assertEqual(1, (yield get1))

    @gen_test
    def test_nonblocking_put_with_getters(self):
        if False:
            for i in range(10):
                print('nop')
        q = queues.Queue()
        get0 = q.get()
        get1 = q.get()
        q.put_nowait(0)
        yield gen.moment
        self.assertEqual(0, (yield get0))
        q.put_nowait(1)
        yield gen.moment
        self.assertEqual(1, (yield get1))

    @gen_test
    def test_blocking_put_wait(self):
        if False:
            return 10
        q = queues.Queue(1)
        q.put_nowait(0)

        def get_and_discard():
            if False:
                for i in range(10):
                    print('nop')
            q.get()
        self.io_loop.call_later(0.01, get_and_discard)
        self.io_loop.call_later(0.02, get_and_discard)
        futures = [q.put(0), q.put(1)]
        self.assertFalse(any((f.done() for f in futures)))
        yield futures

    @gen_test
    def test_put_timeout(self):
        if False:
            while True:
                i = 10
        q = queues.Queue(1)
        q.put_nowait(0)
        put_timeout = q.put(1, timeout=timedelta(seconds=0.01))
        put = q.put(2)
        with self.assertRaises(TimeoutError):
            yield put_timeout
        self.assertEqual(0, q.get_nowait())
        self.assertEqual(2, (yield q.get()))
        yield put

    @gen_test
    def test_put_timeout_preempted(self):
        if False:
            for i in range(10):
                print('nop')
        q = queues.Queue(1)
        q.put_nowait(0)
        put = q.put(1, timeout=timedelta(seconds=0.01))
        q.get()
        yield gen.sleep(0.02)
        yield put

    @gen_test
    def test_put_clears_timed_out_putters(self):
        if False:
            return 10
        q = queues.Queue(1)
        putters = [q.put(i, timedelta(seconds=0.01)) for i in range(10)]
        put = q.put(10)
        self.assertEqual(10, len(q._putters))
        yield gen.sleep(0.02)
        self.assertEqual(10, len(q._putters))
        self.assertFalse(put.done())
        q.put(11)
        self.assertEqual(2, len(q._putters))
        for putter in putters[1:]:
            self.assertRaises(TimeoutError, putter.result)

    @gen_test
    def test_put_clears_timed_out_getters(self):
        if False:
            return 10
        q = queues.Queue()
        getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
        get = asyncio.ensure_future(q.get())
        q.get()
        self.assertEqual(12, len(q._getters))
        yield gen.sleep(0.02)
        self.assertEqual(12, len(q._getters))
        self.assertFalse(get.done())
        q.put(0)
        self.assertEqual(1, len(q._getters))
        self.assertEqual(0, (yield get))
        for getter in getters:
            self.assertRaises(TimeoutError, getter.result)

    @gen_test
    def test_float_maxsize(self):
        if False:
            print('Hello World!')
        q = queues.Queue(maxsize=1.3)
        self.assertTrue(q.empty())
        self.assertFalse(q.full())
        q.put_nowait(0)
        q.put_nowait(1)
        self.assertFalse(q.empty())
        self.assertTrue(q.full())
        self.assertRaises(queues.QueueFull, q.put_nowait, 2)
        self.assertEqual(0, q.get_nowait())
        self.assertFalse(q.empty())
        self.assertFalse(q.full())
        yield q.put(2)
        put = q.put(3)
        self.assertFalse(put.done())
        self.assertEqual(1, (yield q.get()))
        yield put
        self.assertTrue(q.full())

class QueueJoinTest(AsyncTestCase):
    queue_class = queues.Queue

    def test_task_done_underflow(self):
        if False:
            while True:
                i = 10
        q = self.queue_class()
        self.assertRaises(ValueError, q.task_done)

    @gen_test
    def test_task_done(self):
        if False:
            i = 10
            return i + 15
        q = self.queue_class()
        for i in range(100):
            q.put_nowait(i)
        self.accumulator = 0

        @gen.coroutine
        def worker():
            if False:
                print('Hello World!')
            while True:
                item = (yield q.get())
                self.accumulator += item
                q.task_done()
                yield gen.sleep(random() * 0.01)
        worker()
        worker()
        yield q.join()
        self.assertEqual(sum(range(100)), self.accumulator)

    @gen_test
    def test_task_done_delay(self):
        if False:
            print('Hello World!')
        q = self.queue_class()
        q.put_nowait(0)
        join = asyncio.ensure_future(q.join())
        self.assertFalse(join.done())
        yield q.get()
        self.assertFalse(join.done())
        yield gen.moment
        self.assertFalse(join.done())
        q.task_done()
        self.assertTrue(join.done())

    @gen_test
    def test_join_empty_queue(self):
        if False:
            i = 10
            return i + 15
        q = self.queue_class()
        yield q.join()
        yield q.join()

    @gen_test
    def test_join_timeout(self):
        if False:
            i = 10
            return i + 15
        q = self.queue_class()
        q.put(0)
        with self.assertRaises(TimeoutError):
            yield q.join(timeout=timedelta(seconds=0.01))

class PriorityQueueJoinTest(QueueJoinTest):
    queue_class = queues.PriorityQueue

    @gen_test
    def test_order(self):
        if False:
            while True:
                i = 10
        q = self.queue_class(maxsize=2)
        q.put_nowait((1, 'a'))
        q.put_nowait((0, 'b'))
        self.assertTrue(q.full())
        q.put((3, 'c'))
        q.put((2, 'd'))
        self.assertEqual((0, 'b'), q.get_nowait())
        self.assertEqual((1, 'a'), (yield q.get()))
        self.assertEqual((2, 'd'), q.get_nowait())
        self.assertEqual((3, 'c'), (yield q.get()))
        self.assertTrue(q.empty())

class LifoQueueJoinTest(QueueJoinTest):
    queue_class = queues.LifoQueue

    @gen_test
    def test_order(self):
        if False:
            for i in range(10):
                print('nop')
        q = self.queue_class(maxsize=2)
        q.put_nowait(1)
        q.put_nowait(0)
        self.assertTrue(q.full())
        q.put(3)
        q.put(2)
        self.assertEqual(3, q.get_nowait())
        self.assertEqual(2, (yield q.get()))
        self.assertEqual(0, q.get_nowait())
        self.assertEqual(1, (yield q.get()))
        self.assertTrue(q.empty())

class ProducerConsumerTest(AsyncTestCase):

    @gen_test
    def test_producer_consumer(self):
        if False:
            print('Hello World!')
        q = queues.Queue(maxsize=3)
        history = []

        @gen.coroutine
        def consumer():
            if False:
                return 10
            while True:
                history.append((yield q.get()))
                q.task_done()

        @gen.coroutine
        def producer():
            if False:
                for i in range(10):
                    print('nop')
            for item in range(10):
                yield q.put(item)
        consumer()
        yield producer()
        yield q.join()
        self.assertEqual(list(range(10)), history)
if __name__ == '__main__':
    unittest.main()