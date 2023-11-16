"""Unit tests for the retry module."""
import unittest
from parameterized import parameterized
from apache_beam.utils import retry
try:
    from apitools.base.py.exceptions import HttpError
except ImportError:
    HttpError = None

class FakeClock(object):
    """A fake clock object implementing sleep() and recording calls."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.calls = []

    def sleep(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.calls.append(value)

class FakeLogger(object):
    """A fake logger object implementing log() and recording calls."""

    def __init__(self):
        if False:
            return 10
        self.calls = []

    def log(self, message, interval, func_name, exn_name, exn_traceback):
        if False:
            print('Hello World!')
        _ = (interval, exn_traceback)
        self.calls.append((message, func_name, exn_name))

@retry.with_exponential_backoff(clock=FakeClock())
def _test_function(a, b):
    if False:
        while True:
            i = 10
    _ = (a, b)
    raise NotImplementedError

@retry.with_exponential_backoff(initial_delay_secs=0.1, num_retries=1)
def _test_function_with_real_clock(a, b):
    if False:
        return 10
    _ = (a, b)
    raise NotImplementedError

@retry.no_retries
def _test_no_retry_function(a, b):
    if False:
        while True:
            i = 10
    _ = (a, b)
    raise NotImplementedError

class RetryTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.clock = FakeClock()
        self.logger = FakeLogger()
        self.calls = 0

    def permanent_failure(self, a, b):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def transient_failure(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        self.calls += 1
        if self.calls > 4:
            return a + b
        raise NotImplementedError

    def http_error(self, code):
        if False:
            return 10
        if HttpError is None:
            raise RuntimeError('This is not a valid test as GCP is not enabled')
        raise HttpError({'status': str(code)}, '', '')

    def test_with_explicit_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(NotImplementedError, _test_function, 10, b=20)

    def test_with_no_retry_decorator(self):
        if False:
            print('Hello World!')
        self.assertRaises(NotImplementedError, _test_no_retry_function, 1, 2)

    def test_with_real_clock(self):
        if False:
            print('Hello World!')
        self.assertRaises(NotImplementedError, _test_function_with_real_clock, 10, b=20)

    def test_with_default_number_of_retries(self):
        if False:
            print('Hello World!')
        self.assertRaises(NotImplementedError, retry.with_exponential_backoff(clock=self.clock)(self.permanent_failure), 10, b=20)
        self.assertEqual(len(self.clock.calls), 7)

    def test_with_explicit_number_of_retries(self):
        if False:
            while True:
                i = 10
        self.assertRaises(NotImplementedError, retry.with_exponential_backoff(clock=self.clock, num_retries=10)(self.permanent_failure), 10, b=20)
        self.assertEqual(len(self.clock.calls), 10)

    @unittest.skipIf(HttpError is None, 'google-apitools is not installed')
    def test_with_http_error_that_should_not_be_retried(self):
        if False:
            return 10
        self.assertRaises(HttpError, retry.with_exponential_backoff(clock=self.clock, num_retries=10)(self.http_error), 404)
        self.assertEqual(len(self.clock.calls), 0)

    @unittest.skipIf(HttpError is None, 'google-apitools is not installed')
    def test_with_http_error_that_should_be_retried(self):
        if False:
            return 10
        self.assertRaises(HttpError, retry.with_exponential_backoff(clock=self.clock, num_retries=10)(self.http_error), 500)
        self.assertEqual(len(self.clock.calls), 10)

    def test_with_explicit_initial_delay(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(NotImplementedError, retry.with_exponential_backoff(initial_delay_secs=10.0, clock=self.clock, fuzz=False)(self.permanent_failure), 10, b=20)
        self.assertEqual(len(self.clock.calls), 7)
        self.assertEqual(self.clock.calls[0], 10.0)

    @parameterized.expand([(str(i), i) for i in range(0, 1000, 47)])
    def test_with_stop_after_secs(self, _, stop_after_secs):
        if False:
            while True:
                i = 10
        max_delay_secs = 10
        self.assertRaises(NotImplementedError, retry.with_exponential_backoff(num_retries=10000, initial_delay_secs=10.0, clock=self.clock, fuzz=False, max_delay_secs=max_delay_secs, stop_after_secs=stop_after_secs)(self.permanent_failure), 10, b=20)
        total_delay = sum(self.clock.calls)
        self.assertLessEqual(total_delay, stop_after_secs)
        self.assertGreaterEqual(total_delay, stop_after_secs - max_delay_secs)

    def test_log_calls_for_permanent_failure(self):
        if False:
            return 10
        self.assertRaises(NotImplementedError, retry.with_exponential_backoff(clock=self.clock, logger=self.logger.log)(self.permanent_failure), 10, b=20)
        self.assertEqual(len(self.logger.calls), 7)
        for (message, func_name, exn_name) in self.logger.calls:
            self.assertTrue(message.startswith('Retry with exponential backoff:'))
            self.assertEqual(exn_name, 'NotImplementedError\n')
            self.assertEqual(func_name, 'permanent_failure')

    def test_log_calls_for_transient_failure(self):
        if False:
            i = 10
            return i + 15
        result = retry.with_exponential_backoff(clock=self.clock, logger=self.logger.log, fuzz=False)(self.transient_failure)(10, b=20)
        self.assertEqual(result, 30)
        self.assertEqual(len(self.clock.calls), 4)
        self.assertEqual(self.clock.calls, [5.0 * 1, 5.0 * 2, 5.0 * 4, 5.0 * 8])
        self.assertEqual(len(self.logger.calls), 4)
        for (message, func_name, exn_name) in self.logger.calls:
            self.assertTrue(message.startswith('Retry with exponential backoff:'))
            self.assertEqual(exn_name, 'NotImplementedError\n')
            self.assertEqual(func_name, 'transient_failure')

class DummyClass(object):

    def __init__(self, results):
        if False:
            return 10
        self.index = 0
        self.results = results

    @retry.with_exponential_backoff(num_retries=2, initial_delay_secs=0.1)
    def func(self):
        if False:
            for i in range(10):
                print('nop')
        self.index += 1
        if self.index > len(self.results) or self.results[self.index - 1] == 'Error':
            raise ValueError('Error')
        return self.results[self.index - 1]

class RetryStateTest(unittest.TestCase):
    """The test_two_failures and test_single_failure would fail if we have
  any shared state for the retry decorator. This test tries to prevent a bug we
  found where the state in the decorator was shared across objects and retries
  were not available correctly.

  The test_call_two_objects would test this inside the same test.
  """

    def test_two_failures(self):
        if False:
            print('Hello World!')
        dummy = DummyClass(['Error', 'Error', 'Success'])
        dummy.func()
        self.assertEqual(3, dummy.index)

    def test_single_failure(self):
        if False:
            while True:
                i = 10
        dummy = DummyClass(['Error', 'Success'])
        dummy.func()
        self.assertEqual(2, dummy.index)

    def test_call_two_objects(self):
        if False:
            return 10
        dummy = DummyClass(['Error', 'Error', 'Success'])
        dummy.func()
        self.assertEqual(3, dummy.index)
        dummy2 = DummyClass(['Error', 'Success'])
        dummy2.func()
        self.assertEqual(2, dummy2.index)
if __name__ == '__main__':
    unittest.main()