from gevent import monkey
monkey.patch_all()
from gevent import queue as Queue
import threading
import time
import unittest
QUEUE_SIZE = 5

class _TriggerThread(threading.Thread):

    def __init__(self, fn, args):
        if False:
            i = 10
            return i + 15
        self.fn = fn
        self.args = args
        from gevent.event import Event
        self.startedEvent = Event()
        threading.Thread.__init__(self)

    def run(self):
        if False:
            i = 10
            return i + 15
        time.sleep(0.01)
        self.startedEvent.set()
        self.fn(*self.args)

class BlockingTestMixin(object):

    def do_blocking_test(self, block_func, block_args, trigger_func, trigger_args):
        if False:
            i = 10
            return i + 15
        self.t = _TriggerThread(trigger_func, trigger_args)
        self.t.start()
        self.result = block_func(*block_args)
        if not self.t.startedEvent.isSet():
            self.fail("blocking function '%r' appeared not to block" % block_func)
        self.t.join(10)
        if self.t.is_alive():
            self.fail("trigger function '%r' appeared to not return" % trigger_func)
        return self.result

    def do_exceptional_blocking_test(self, block_func, block_args, trigger_func, trigger_args, expected_exception_class):
        if False:
            return 10
        self.t = _TriggerThread(trigger_func, trigger_args)
        self.t.start()
        try:
            with self.assertRaises(expected_exception_class):
                block_func(*block_args)
        finally:
            self.t.join(10)
            if self.t.is_alive():
                self.fail("trigger function '%r' appeared to not return" % trigger_func)
            if not self.t.startedEvent.isSet():
                self.fail('trigger thread ended but event never set')

class BaseQueueTest(unittest.TestCase, BlockingTestMixin):
    type2test = Queue.Queue

    def setUp(self):
        if False:
            while True:
                i = 10
        self.cum = 0
        self.cumlock = threading.Lock()

    def simple_queue_test(self, q):
        if False:
            for i in range(10):
                print('nop')
        if not q.empty():
            raise RuntimeError('Call this function with an empty queue')
        q.put(111)
        q.put(333)
        q.put(222)
        q.put(444)
        target_first_items = dict(Queue=111, LifoQueue=444, PriorityQueue=111)
        actual_first_item = (q.peek(), q.get())
        self.assertEqual(actual_first_item, (target_first_items[q.__class__.__name__], target_first_items[q.__class__.__name__]), 'q.peek() and q.get() are not equal!')
        target_order = dict(Queue=[333, 222, 444], LifoQueue=[222, 333, 111], PriorityQueue=[222, 333, 444])
        actual_order = [q.get(), q.get(), q.get()]
        self.assertEqual(actual_order, target_order[q.__class__.__name__], "Didn't seem to queue the correct data!")
        for i in range(QUEUE_SIZE - 1):
            q.put(i)
            self.assertFalse(q.empty(), 'Queue should not be empty')
        self.assertFalse(q.full(), 'Queue should not be full')
        q.put(999)
        self.assertTrue(q.full(), 'Queue should be full')
        try:
            q.put(888, block=0)
            self.fail("Didn't appear to block with a full queue")
        except Queue.Full:
            pass
        try:
            q.put(888, timeout=0.01)
            self.fail("Didn't appear to time-out with a full queue")
        except Queue.Full:
            pass
        self.assertEqual(q.qsize(), QUEUE_SIZE)
        self.do_blocking_test(q.put, (888,), q.get, ())
        self.do_blocking_test(q.put, (888, True, 10), q.get, ())
        for i in range(QUEUE_SIZE):
            q.get()
        self.assertTrue(q.empty(), 'Queue should be empty')
        try:
            q.get(block=0)
            self.fail("Didn't appear to block with an empty queue")
        except Queue.Empty:
            pass
        try:
            q.get(timeout=0.01)
            self.fail("Didn't appear to time-out with an empty queue")
        except Queue.Empty:
            pass
        self.do_blocking_test(q.get, (), q.put, ('empty',))
        self.do_blocking_test(q.get, (True, 10), q.put, ('empty',))

    def worker(self, q):
        if False:
            while True:
                i = 10
        while True:
            x = q.get()
            if x is None:
                q.task_done()
                return
            self.cum += x
            q.task_done()

    def queue_join_test(self, q):
        if False:
            return 10
        self.cum = 0
        for i in (0, 1):
            threading.Thread(target=self.worker, args=(q,)).start()
        for i in range(100):
            q.put(i)
        q.join()
        self.assertEqual(self.cum, sum(range(100)), 'q.join() did not block until all tasks were done')
        for i in (0, 1):
            q.put(None)
        q.join()

    def test_queue_task_done(self):
        if False:
            for i in range(10):
                print('nop')
        q = Queue.JoinableQueue()
        try:
            q.task_done()
        except ValueError:
            pass
        else:
            self.fail('Did not detect task count going negative')

    def test_queue_join(self):
        if False:
            return 10
        q = Queue.JoinableQueue()
        self.queue_join_test(q)
        self.queue_join_test(q)
        try:
            q.task_done()
        except ValueError:
            pass
        else:
            self.fail('Did not detect task count going negative')

    def test_queue_task_done_with_items(self):
        if False:
            for i in range(10):
                print('nop')
        l = [1, 2, 3]
        q = Queue.JoinableQueue(items=l)
        for i in l:
            self.assertFalse(q.join(timeout=0.001))
            self.assertEqual(i, q.get())
            q.task_done()
        try:
            q.task_done()
        except ValueError:
            pass
        else:
            self.fail('Did not detect task count going negative')
        self.assertTrue(q.join(timeout=0.001))

    def test_simple_queue(self):
        if False:
            while True:
                i = 10
        q = self.type2test(QUEUE_SIZE)
        self.simple_queue_test(q)
        self.simple_queue_test(q)

class LifoQueueTest(BaseQueueTest):
    type2test = Queue.LifoQueue

class PriorityQueueTest(BaseQueueTest):
    type2test = Queue.PriorityQueue

    def test__init(self):
        if False:
            i = 10
            return i + 15
        item1 = (2, 'b')
        item2 = (1, 'a')
        q = self.type2test(items=[item1, item2])
        self.assertTupleEqual(item2, q.get_nowait())
        self.assertTupleEqual(item1, q.get_nowait())

class FailingQueueException(Exception):
    pass

class FailingQueue(Queue.Queue):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        self.fail_next_put = False
        self.fail_next_get = False
        Queue.Queue.__init__(self, *args)

    def _put(self, item):
        if False:
            return 10
        if self.fail_next_put:
            self.fail_next_put = False
            raise FailingQueueException('You Lose')
        return Queue.Queue._put(self, item)

    def _get(self):
        if False:
            return 10
        if self.fail_next_get:
            self.fail_next_get = False
            raise FailingQueueException('You Lose')
        return Queue.Queue._get(self)

class FailingQueueTest(unittest.TestCase, BlockingTestMixin):

    def failing_queue_test(self, q):
        if False:
            for i in range(10):
                print('nop')
        if not q.empty():
            raise RuntimeError('Call this function with an empty queue')
        for i in range(QUEUE_SIZE - 1):
            q.put(i)
        q.fail_next_put = True
        with self.assertRaises(FailingQueueException):
            q.put('oops', block=0)
        q.fail_next_put = True
        with self.assertRaises(FailingQueueException):
            q.put('oops', timeout=0.1)
        q.put(999)
        self.assertTrue(q.full(), 'Queue should be full')
        q.fail_next_put = True
        with self.assertRaises(FailingQueueException):
            self.do_blocking_test(q.put, (888,), q.get, ())
        q.put(999)
        q.fail_next_put = True
        self.do_exceptional_blocking_test(q.put, (888, True, 10), q.get, (), FailingQueueException)
        q.put(999)
        self.assertTrue(q.full(), 'Queue should be full')
        q.get()
        self.assertFalse(q.full(), 'Queue should not be full')
        q.put(999)
        self.assertTrue(q.full(), 'Queue should be full')
        self.do_blocking_test(q.put, (888,), q.get, ())
        for i in range(QUEUE_SIZE):
            q.get()
        self.assertTrue(q.empty(), 'Queue should be empty')
        q.put('first')
        q.fail_next_get = True
        with self.assertRaises(FailingQueueException):
            q.get()
        self.assertFalse(q.empty(), 'Queue should not be empty')
        q.fail_next_get = True
        with self.assertRaises(FailingQueueException):
            q.get(timeout=0.1)
        self.assertFalse(q.empty(), 'Queue should not be empty')
        q.get()
        self.assertTrue(q.empty(), 'Queue should be empty')
        q.fail_next_get = True
        self.do_exceptional_blocking_test(q.get, (), q.put, ('empty',), FailingQueueException)
        self.assertFalse(q.empty(), 'Queue should not be empty')
        q.get()
        self.assertTrue(q.empty(), 'Queue should be empty')

    def test_failing_queue(self):
        if False:
            return 10
        q = FailingQueue(QUEUE_SIZE)
        self.failing_queue_test(q)
        self.failing_queue_test(q)
if __name__ == '__main__':
    unittest.main()