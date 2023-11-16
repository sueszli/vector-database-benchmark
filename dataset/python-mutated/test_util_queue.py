import threading
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.util.backoff import BackoffTimeoutExceededError
from buildbot.util.queue import ConnectableThreadQueue

class FakeConnection:
    pass

class TestableConnectableThreadQueue(ConnectableThreadQueue):

    def __init__(self, case, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.case = case
        self.create_connection_called_count = 0
        self.close_connection_called_count = 0
        self._test_conn = None

    def create_connection(self):
        if False:
            for i in range(10):
                print('nop')
        self.case.assertTrue(self.connecting)
        self.create_connection_called_count += 1
        self.case.assertIsNone(self._test_conn)
        self._test_conn = FakeConnection()
        return self._test_conn

    def on_close_connection(self, conn):
        if False:
            while True:
                i = 10
        self.case.assertIs(conn, self._test_conn)
        self._test_conn = None
        self.close_connection()

    def close_connection(self):
        if False:
            while True:
                i = 10
        self.case.assertFalse(self.connecting)
        self._test_conn = None
        self.close_connection_called_count += 1
        super().close_connection()

class TestException(Exception):
    pass

class TestConnectableThreadQueue(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.queue = TestableConnectableThreadQueue(self, connect_backoff_start_seconds=0, connect_backoff_multiplier=0, connect_backoff_max_wait_seconds=0)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.join_queue()

    def join_queue(self, connection_called_count=None):
        if False:
            while True:
                i = 10
        self.queue.join(timeout=1)
        if self.queue.is_alive():
            raise AssertionError('Thread is still alive')
        if connection_called_count is not None:
            self.assertEqual(self.queue.create_connection_called_count, connection_called_count)
            self.assertEqual(self.queue.close_connection_called_count, connection_called_count)

    def test_no_work(self):
        if False:
            print('Hello World!')
        self.join_queue(0)

    @defer.inlineCallbacks
    def test_single_item_called(self):
        if False:
            print('Hello World!')

        def work(conn, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            self.assertIs(conn, self.queue.conn)
            self.assertEqual(args, ('arg',))
            self.assertEqual(kwargs, {'kwarg': 'kwvalue'})
            return 'work_result'
        result = (yield self.queue.execute_in_thread(work, 'arg', kwarg='kwvalue'))
        self.assertEqual(result, 'work_result')
        self.join_queue(1)

    @defer.inlineCallbacks
    def test_single_item_called_exception(self):
        if False:
            return 10

        def work(conn):
            if False:
                return 10
            raise TestException()
        with self.assertRaises(TestException):
            yield self.queue.execute_in_thread(work)
        self.join_queue(1)

    @defer.inlineCallbacks
    def test_exception_does_not_break_further_work(self):
        if False:
            for i in range(10):
                print('nop')

        def work_exception(conn):
            if False:
                while True:
                    i = 10
            raise TestException()

        def work_success(conn):
            if False:
                return 10
            return 'work_result'
        with self.assertRaises(TestException):
            yield self.queue.execute_in_thread(work_exception)
        result = (yield self.queue.execute_in_thread(work_success))
        self.assertEqual(result, 'work_result')
        self.join_queue(1)

    @defer.inlineCallbacks
    def test_single_item_called_disconnect(self):
        if False:
            for i in range(10):
                print('nop')

        def work(conn):
            if False:
                while True:
                    i = 10
            pass
        yield self.queue.execute_in_thread(work)
        self.queue.close_connection()
        yield self.queue.execute_in_thread(work)
        self.join_queue(2)

    @defer.inlineCallbacks
    def test_many_items_called_in_order(self):
        if False:
            return 10
        self.expected_work_index = 0

        def work(conn, work_index):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.expected_work_index, work_index)
            self.expected_work_index = work_index + 1
            return work_index
        work_deferreds = [self.queue.execute_in_thread(work, i) for i in range(0, 100)]
        for (i, d) in enumerate(work_deferreds):
            self.assertEqual((yield d), i)
        self.join_queue(1)

class FailingConnectableThreadQueue(ConnectableThreadQueue):

    def __init__(self, case, lock, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.case = case
        self.lock = lock
        self.create_connection_called_count = 0

    def on_close_connection(self, conn):
        if False:
            for i in range(10):
                print('nop')
        raise AssertionError('on_close_connection should not have been called')

    def close_connection(self):
        if False:
            print('Hello World!')
        raise AssertionError('close_connection should not have been called')

    def _drain_queue_with_exception(self, e):
        if False:
            print('Hello World!')
        with self.lock:
            return super()._drain_queue_with_exception(e)

class ThrowingConnectableThreadQueue(FailingConnectableThreadQueue):

    def create_connection(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            self.create_connection_called_count += 1
            self.case.assertTrue(self.connecting)
            raise TestException()

class NoneReturningConnectableThreadQueue(FailingConnectableThreadQueue):

    def create_connection(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            self.create_connection_called_count += 1
            self.case.assertTrue(self.connecting)
            return None

class ConnectionErrorTests:

    def setUp(self):
        if False:
            while True:
                i = 10
        self.lock = threading.Lock()
        self.queue = self.QueueClass(self, self.lock, connect_backoff_start_seconds=0.001, connect_backoff_multiplier=1, connect_backoff_max_wait_seconds=0.0039)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.queue.join(timeout=1)
        if self.queue.is_alive():
            raise AssertionError('Thread is still alive')

    @defer.inlineCallbacks
    def test_resets_after_reject(self):
        if False:
            for i in range(10):
                print('nop')

        def work(conn):
            if False:
                i = 10
                return i + 15
            raise AssertionError('work should not be executed')
        with self.lock:
            d = self.queue.execute_in_thread(work)
        with self.assertRaises(BackoffTimeoutExceededError):
            yield d
        self.assertEqual(self.queue.create_connection_called_count, 5)
        with self.lock:
            d = self.queue.execute_in_thread(work)
        with self.assertRaises(BackoffTimeoutExceededError):
            yield d
        self.assertEqual(self.queue.create_connection_called_count, 10)
        self.flushLoggedErrors(TestException)

    @defer.inlineCallbacks
    def test_multiple_work_rejected(self):
        if False:
            return 10

        def work(conn):
            if False:
                return 10
            raise AssertionError('work should not be executed')
        with self.lock:
            d1 = self.queue.execute_in_thread(work)
            d2 = self.queue.execute_in_thread(work)
            d3 = self.queue.execute_in_thread(work)
        with self.assertRaises(BackoffTimeoutExceededError):
            yield d1
        with self.assertRaises(BackoffTimeoutExceededError):
            yield d2
        with self.assertRaises(BackoffTimeoutExceededError):
            yield d3
        self.assertEqual(self.queue.create_connection_called_count, 5)
        self.flushLoggedErrors(TestException)

class TestConnectionErrorThrow(ConnectionErrorTests, unittest.TestCase):
    QueueClass = ThrowingConnectableThreadQueue

class TestConnectionErrorReturnNone(ConnectionErrorTests, unittest.TestCase):
    QueueClass = NoneReturningConnectableThreadQueue