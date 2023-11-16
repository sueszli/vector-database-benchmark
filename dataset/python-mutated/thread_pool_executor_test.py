"""Unit tests for UnboundedThreadPoolExecutor."""
import itertools
import threading
import time
import traceback
import unittest
from apache_beam.utils import thread_pool_executor
from apache_beam.utils.thread_pool_executor import UnboundedThreadPoolExecutor

class UnboundedThreadPoolExecutorTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._lock = threading.Lock()
        self._worker_idents = []

    def append_and_sleep(self, sleep_time):
        if False:
            return 10
        with self._lock:
            self._worker_idents.append(threading.current_thread().ident)
        time.sleep(sleep_time)

    def raise_error(self, message):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError(message)

    def test_shutdown_with_no_workers(self):
        if False:
            i = 10
            return i + 15
        with UnboundedThreadPoolExecutor():
            pass

    def test_shutdown_with_fast_workers(self):
        if False:
            while True:
                i = 10
        futures = []
        with UnboundedThreadPoolExecutor() as executor:
            for _ in range(0, 5):
                futures.append(executor.submit(self.append_and_sleep, 0.01))
        for future in futures:
            future.result(timeout=10)
        with self._lock:
            self.assertEqual(5, len(self._worker_idents))

    def test_shutdown_with_slow_workers(self):
        if False:
            while True:
                i = 10
        futures = []
        with UnboundedThreadPoolExecutor() as executor:
            for _ in range(0, 5):
                futures.append(executor.submit(self.append_and_sleep, 1))
        for future in futures:
            future.result(timeout=10)
        with self._lock:
            self.assertEqual(5, len(self._worker_idents))

    def test_worker_reuse(self):
        if False:
            return 10
        futures = []
        with UnboundedThreadPoolExecutor() as executor:
            for _ in range(0, 5):
                futures.append(executor.submit(self.append_and_sleep, 0.01))
            time.sleep(3)
            for _ in range(0, 5):
                futures.append(executor.submit(self.append_and_sleep, 0.01))
        for future in futures:
            future.result(timeout=10)
        with self._lock:
            self.assertEqual(10, len(self._worker_idents))
            self.assertTrue(len(set(self._worker_idents)) < 10)

    def test_exception_propagation(self):
        if False:
            while True:
                i = 10
        with UnboundedThreadPoolExecutor() as executor:
            future = executor.submit(self.raise_error, 'footest')
        try:
            future.result()
        except Exception:
            message = traceback.format_exc()
        else:
            raise AssertionError('expected exception not raised')
        self.assertIn('footest', message)
        self.assertIn('raise_error', message)

    def test_map(self):
        if False:
            for i in range(10):
                print('nop')
        with UnboundedThreadPoolExecutor() as executor:
            executor.map(self.append_and_sleep, itertools.repeat(0.01, 5))
        with self._lock:
            self.assertEqual(5, len(self._worker_idents))

    def test_shared_shutdown_does_nothing(self):
        if False:
            print('Hello World!')
        thread_pool_executor.shared_unbounded_instance().shutdown()
        futures = []
        with thread_pool_executor.shared_unbounded_instance() as executor:
            for _ in range(0, 5):
                futures.append(executor.submit(self.append_and_sleep, 0.01))
        for future in futures:
            future.result(timeout=10)
        with self._lock:
            self.assertEqual(5, len(self._worker_idents))
if __name__ == '__main__':
    unittest.main()