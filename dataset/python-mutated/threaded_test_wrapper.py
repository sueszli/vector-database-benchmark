"""
Implements run_in_thread_with_timeout decorator for running tests that might
deadlock.

"""
from __future__ import print_function
import functools
import os
import sys
import threading
import traceback
import unittest
MODULE_PID = os.getpid()
DEFAULT_TEST_TIMEOUT = 15

def create_run_in_thread_decorator(test_timeout=None):
    if False:
        while True:
            i = 10
    'Create a decorator that will run the decorated method in a thread via\n    `_ThreadedTestWrapper` and return the value that is returned by the\n    given function, unless it exits with exception or times out, in which\n    case AssertionError will be raised\n\n    :param int | float | None test_timeout: maximum number of seconds to wait\n        for test to complete. If None, `DEFAULT_TEST_TIMEOUT` will be used.\n        NOTE: we handle default this way to facilitate patching of the timeout\n        in our self-tests.\n    :return: decorator\n    '

    def run_in_thread_with_timeout_decorator(fun):
        if False:
            for i in range(10):
                print('nop')
        'Create a wrapper that will run the decorated method in a thread via\n        `_ThreadedTestWrapper` and return the value that is returned by the\n        given function, unless it exits with exception or times out, in which\n        case AssertionError will be raised\n\n        :param fun: function to run in thread\n        :return: wrapper function\n        '

        @functools.wraps(fun)
        def run_in_thread_with_timeout_wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            '\n\n            :param args: positional args to pass to wrapped function\n            :param kwargs: keyword args to pass to wrapped function\n            :return: value returned by the function, unless it exits with\n                exception or times out\n            :raises AssertionError: if wrapped function exits with exception or\n                times out\n            '
            runner = _ThreadedTestWrapper(functools.partial(fun, *args, **kwargs), test_timeout)
            return runner.kick_off()
        return run_in_thread_with_timeout_wrapper
    return run_in_thread_with_timeout_decorator
run_in_thread_with_timeout = create_run_in_thread_decorator()

class _ThreadedTestWrapper(object):
    """Runs user's function in a thread. Then wait on the
    thread to terminate up to the given `test_timeout` seconds, raising
    `AssertionError` if user's function exits with exception or times out.

    """
    _stderr = sys.stderr

    def __init__(self, fun, test_timeout):
        if False:
            return 10
        '\n        :param callable fun: the function to run in thread, no args.\n        :param int | float test_timeout: maximum number of seconds to wait for\n            thread to exit.\n\n        '
        self._fun = fun
        if test_timeout is None:
            self._test_timeout = DEFAULT_TEST_TIMEOUT
        else:
            self._test_timeout = test_timeout
        self._stderr = self._stderr
        self._fun_result = None
        self._exc_info = None

    def kick_off(self):
        if False:
            print('Hello World!')
        "Run user's function in a thread. Then wait on the\n        thread to terminate up to self._test_timeout seconds, raising\n        `AssertionError` if user's function exits with exception or times out.\n\n        :return: the value returned by function if function exited without\n            exception and didn't time out\n        :raises AssertionError: if user's function timed out or exited with\n            exception.\n        "
        try:
            runner = threading.Thread(target=self._thread_entry)
            runner.daemon = True
            runner.start()
            runner.join(self._test_timeout)
            if runner.is_alive():
                raise AssertionError('The test timed out.')
            if self._exc_info is not None:
                if isinstance(self._exc_info[1], unittest.SkipTest):
                    raise self._exc_info[1]
                raise AssertionError(self._exc_info_to_str(self._exc_info))
            return self._fun_result
        finally:
            self._exc_info = None
            self._fun = None

    def _thread_entry(self):
        if False:
            i = 10
            return i + 15
        "Our test-execution thread entry point that calls the test's `start()`\n        method.\n\n        Here, we catch all exceptions from `start()`, save the `exc_info` for\n        processing by `_kick_off()`, and print the stack trace to `sys.stderr`.\n        "
        try:
            self._fun_result = self._fun()
        except:
            self._exc_info = sys.exc_info()
            del self._fun_result
            if not isinstance(self._exc_info[1], unittest.SkipTest):
                print('ERROR start() of test {} failed:\n{}'.format(self, self._exc_info_to_str(self._exc_info)), end='', file=self._stderr)

    @staticmethod
    def _exc_info_to_str(exc_info):
        if False:
            i = 10
            return i + 15
        'Convenience method for converting the value returned by\n        `sys.exc_info()` to a string.\n\n        :param tuple exc_info: Value returned by `sys.exc_info()`.\n        :return: A string representation of the given `exc_info`.\n        :rtype: str\n        '
        return ''.join(traceback.format_exception(*exc_info))