import datetime
import logging
import re
import sys
import time
import typing
import unittest
import warnings
from contextlib import contextmanager
from copy import copy
from fractions import Fraction
import pytest
import tenacity
from tenacity import RetryCallState, RetryError, Retrying, retry
_unset = object()

def _make_unset_exception(func_name, **kwargs):
    if False:
        while True:
            i = 10
    missing = []
    for (k, v) in kwargs.items():
        if v is _unset:
            missing.append(k)
    missing_str = ', '.join((repr(s) for s in missing))
    return TypeError(func_name + ' func missing parameters: ' + missing_str)

def _set_delay_since_start(retry_state, delay):
    if False:
        for i in range(10):
            print('nop')
    retry_state.start_time = Fraction(retry_state.start_time)
    retry_state.outcome_timestamp = retry_state.start_time + Fraction(delay)
    assert retry_state.seconds_since_start == delay

def make_retry_state(previous_attempt_number, delay_since_first_attempt, last_result=None):
    if False:
        return 10
    'Construct RetryCallState for given attempt number & delay.\n\n    Only used in testing and thus is extra careful about timestamp arithmetics.\n    '
    required_parameter_unset = previous_attempt_number is _unset or delay_since_first_attempt is _unset
    if required_parameter_unset:
        raise _make_unset_exception('wait/stop', previous_attempt_number=previous_attempt_number, delay_since_first_attempt=delay_since_first_attempt)
    retry_state = RetryCallState(None, None, (), {})
    retry_state.attempt_number = previous_attempt_number
    if last_result is not None:
        retry_state.outcome = last_result
    else:
        retry_state.set_result(None)
    _set_delay_since_start(retry_state, delay_since_first_attempt)
    return retry_state

class TestBase(unittest.TestCase):

    def test_retrying_repr(self):
        if False:
            print('Hello World!')

        class ConcreteRetrying(tenacity.BaseRetrying):

            def __call__(self, fn, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pass
        repr(ConcreteRetrying())

    def test_callstate_repr(self):
        if False:
            while True:
                i = 10
        rs = RetryCallState(None, None, (), {})
        rs.idle_for = 1.1111111
        assert repr(rs).endswith('attempt #1; slept for 1.11; last result: none yet>')
        rs = make_retry_state(2, 5)
        assert repr(rs).endswith('attempt #2; slept for 0.0; last result: returned None>')
        rs = make_retry_state(0, 0, last_result=tenacity.Future.construct(1, ValueError('aaa'), True))
        assert repr(rs).endswith('attempt #0; slept for 0.0; last result: failed (ValueError aaa)>')

class TestStopConditions(unittest.TestCase):

    def test_never_stop(self):
        if False:
            print('Hello World!')
        r = Retrying()
        self.assertFalse(r.stop(make_retry_state(3, 6546)))

    def test_stop_any(self):
        if False:
            print('Hello World!')
        stop = tenacity.stop_any(tenacity.stop_after_delay(1), tenacity.stop_after_attempt(4))

        def s(*args):
            if False:
                for i in range(10):
                    print('nop')
            return stop(make_retry_state(*args))
        self.assertFalse(s(1, 0.1))
        self.assertFalse(s(2, 0.2))
        self.assertFalse(s(2, 0.8))
        self.assertTrue(s(4, 0.8))
        self.assertTrue(s(3, 1.8))
        self.assertTrue(s(4, 1.8))

    def test_stop_all(self):
        if False:
            print('Hello World!')
        stop = tenacity.stop_all(tenacity.stop_after_delay(1), tenacity.stop_after_attempt(4))

        def s(*args):
            if False:
                for i in range(10):
                    print('nop')
            return stop(make_retry_state(*args))
        self.assertFalse(s(1, 0.1))
        self.assertFalse(s(2, 0.2))
        self.assertFalse(s(2, 0.8))
        self.assertFalse(s(4, 0.8))
        self.assertFalse(s(3, 1.8))
        self.assertTrue(s(4, 1.8))

    def test_stop_or(self):
        if False:
            i = 10
            return i + 15
        stop = tenacity.stop_after_delay(1) | tenacity.stop_after_attempt(4)

        def s(*args):
            if False:
                return 10
            return stop(make_retry_state(*args))
        self.assertFalse(s(1, 0.1))
        self.assertFalse(s(2, 0.2))
        self.assertFalse(s(2, 0.8))
        self.assertTrue(s(4, 0.8))
        self.assertTrue(s(3, 1.8))
        self.assertTrue(s(4, 1.8))

    def test_stop_and(self):
        if False:
            while True:
                i = 10
        stop = tenacity.stop_after_delay(1) & tenacity.stop_after_attempt(4)

        def s(*args):
            if False:
                i = 10
                return i + 15
            return stop(make_retry_state(*args))
        self.assertFalse(s(1, 0.1))
        self.assertFalse(s(2, 0.2))
        self.assertFalse(s(2, 0.8))
        self.assertFalse(s(4, 0.8))
        self.assertFalse(s(3, 1.8))
        self.assertTrue(s(4, 1.8))

    def test_stop_after_attempt(self):
        if False:
            while True:
                i = 10
        r = Retrying(stop=tenacity.stop_after_attempt(3))
        self.assertFalse(r.stop(make_retry_state(2, 6546)))
        self.assertTrue(r.stop(make_retry_state(3, 6546)))
        self.assertTrue(r.stop(make_retry_state(4, 6546)))

    def test_stop_after_delay(self):
        if False:
            for i in range(10):
                print('nop')
        for delay in (1, datetime.timedelta(seconds=1)):
            with self.subTest():
                r = Retrying(stop=tenacity.stop_after_delay(delay))
                self.assertFalse(r.stop(make_retry_state(2, 0.999)))
                self.assertTrue(r.stop(make_retry_state(2, 1)))
                self.assertTrue(r.stop(make_retry_state(2, 1.001)))

    def test_legacy_explicit_stop_type(self):
        if False:
            i = 10
            return i + 15
        Retrying(stop='stop_after_attempt')

    def test_stop_func_with_retry_state(self):
        if False:
            while True:
                i = 10

        def stop_func(retry_state):
            if False:
                while True:
                    i = 10
            rs = retry_state
            return rs.attempt_number == rs.seconds_since_start
        r = Retrying(stop=stop_func)
        self.assertFalse(r.stop(make_retry_state(1, 3)))
        self.assertFalse(r.stop(make_retry_state(100, 99)))
        self.assertTrue(r.stop(make_retry_state(101, 101)))

class TestWaitConditions(unittest.TestCase):

    def test_no_sleep(self):
        if False:
            i = 10
            return i + 15
        r = Retrying()
        self.assertEqual(0, r.wait(make_retry_state(18, 9879)))

    def test_fixed_sleep(self):
        if False:
            i = 10
            return i + 15
        for wait in (1, datetime.timedelta(seconds=1)):
            with self.subTest():
                r = Retrying(wait=tenacity.wait_fixed(wait))
                self.assertEqual(1, r.wait(make_retry_state(12, 6546)))

    def test_incrementing_sleep(self):
        if False:
            for i in range(10):
                print('nop')
        for (start, increment) in ((500, 100), (datetime.timedelta(seconds=500), datetime.timedelta(seconds=100))):
            with self.subTest():
                r = Retrying(wait=tenacity.wait_incrementing(start=start, increment=increment))
                self.assertEqual(500, r.wait(make_retry_state(1, 6546)))
                self.assertEqual(600, r.wait(make_retry_state(2, 6546)))
                self.assertEqual(700, r.wait(make_retry_state(3, 6546)))

    def test_random_sleep(self):
        if False:
            return 10
        for (min_, max_) in ((1, 20), (datetime.timedelta(seconds=1), datetime.timedelta(seconds=20))):
            with self.subTest():
                r = Retrying(wait=tenacity.wait_random(min=min_, max=max_))
                times = set()
                for _ in range(1000):
                    times.add(r.wait(make_retry_state(1, 6546)))
                self.assertTrue(len(times) > 1)
                for t in times:
                    self.assertTrue(t >= 1)
                    self.assertTrue(t < 20)

    def test_random_sleep_withoutmin_(self):
        if False:
            while True:
                i = 10
        r = Retrying(wait=tenacity.wait_random(max=2))
        times = set()
        times.add(r.wait(make_retry_state(1, 6546)))
        times.add(r.wait(make_retry_state(1, 6546)))
        times.add(r.wait(make_retry_state(1, 6546)))
        times.add(r.wait(make_retry_state(1, 6546)))
        self.assertTrue(len(times) > 1)
        for t in times:
            self.assertTrue(t >= 0)
            self.assertTrue(t <= 2)

    def test_exponential(self):
        if False:
            return 10
        r = Retrying(wait=tenacity.wait_exponential())
        self.assertEqual(r.wait(make_retry_state(1, 0)), 1)
        self.assertEqual(r.wait(make_retry_state(2, 0)), 2)
        self.assertEqual(r.wait(make_retry_state(3, 0)), 4)
        self.assertEqual(r.wait(make_retry_state(4, 0)), 8)
        self.assertEqual(r.wait(make_retry_state(5, 0)), 16)
        self.assertEqual(r.wait(make_retry_state(6, 0)), 32)
        self.assertEqual(r.wait(make_retry_state(7, 0)), 64)
        self.assertEqual(r.wait(make_retry_state(8, 0)), 128)

    def test_exponential_with_max_wait(self):
        if False:
            return 10
        r = Retrying(wait=tenacity.wait_exponential(max=40))
        self.assertEqual(r.wait(make_retry_state(1, 0)), 1)
        self.assertEqual(r.wait(make_retry_state(2, 0)), 2)
        self.assertEqual(r.wait(make_retry_state(3, 0)), 4)
        self.assertEqual(r.wait(make_retry_state(4, 0)), 8)
        self.assertEqual(r.wait(make_retry_state(5, 0)), 16)
        self.assertEqual(r.wait(make_retry_state(6, 0)), 32)
        self.assertEqual(r.wait(make_retry_state(7, 0)), 40)
        self.assertEqual(r.wait(make_retry_state(8, 0)), 40)
        self.assertEqual(r.wait(make_retry_state(50, 0)), 40)

    def test_exponential_with_min_wait(self):
        if False:
            i = 10
            return i + 15
        r = Retrying(wait=tenacity.wait_exponential(min=20))
        self.assertEqual(r.wait(make_retry_state(1, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(2, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(3, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(4, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(5, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(6, 0)), 32)
        self.assertEqual(r.wait(make_retry_state(7, 0)), 64)
        self.assertEqual(r.wait(make_retry_state(8, 0)), 128)
        self.assertEqual(r.wait(make_retry_state(20, 0)), 524288)

    def test_exponential_with_max_wait_and_multiplier(self):
        if False:
            print('Hello World!')
        r = Retrying(wait=tenacity.wait_exponential(max=50, multiplier=1))
        self.assertEqual(r.wait(make_retry_state(1, 0)), 1)
        self.assertEqual(r.wait(make_retry_state(2, 0)), 2)
        self.assertEqual(r.wait(make_retry_state(3, 0)), 4)
        self.assertEqual(r.wait(make_retry_state(4, 0)), 8)
        self.assertEqual(r.wait(make_retry_state(5, 0)), 16)
        self.assertEqual(r.wait(make_retry_state(6, 0)), 32)
        self.assertEqual(r.wait(make_retry_state(7, 0)), 50)
        self.assertEqual(r.wait(make_retry_state(8, 0)), 50)
        self.assertEqual(r.wait(make_retry_state(50, 0)), 50)

    def test_exponential_with_min_wait_and_multiplier(self):
        if False:
            i = 10
            return i + 15
        r = Retrying(wait=tenacity.wait_exponential(min=20, multiplier=2))
        self.assertEqual(r.wait(make_retry_state(1, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(2, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(3, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(4, 0)), 20)
        self.assertEqual(r.wait(make_retry_state(5, 0)), 32)
        self.assertEqual(r.wait(make_retry_state(6, 0)), 64)
        self.assertEqual(r.wait(make_retry_state(7, 0)), 128)
        self.assertEqual(r.wait(make_retry_state(8, 0)), 256)
        self.assertEqual(r.wait(make_retry_state(20, 0)), 1048576)

    def test_exponential_with_min_wait_andmax__wait(self):
        if False:
            i = 10
            return i + 15
        for (min_, max_) in ((10, 100), (datetime.timedelta(seconds=10), datetime.timedelta(seconds=100))):
            with self.subTest():
                r = Retrying(wait=tenacity.wait_exponential(min=min_, max=max_))
                self.assertEqual(r.wait(make_retry_state(1, 0)), 10)
                self.assertEqual(r.wait(make_retry_state(2, 0)), 10)
                self.assertEqual(r.wait(make_retry_state(3, 0)), 10)
                self.assertEqual(r.wait(make_retry_state(4, 0)), 10)
                self.assertEqual(r.wait(make_retry_state(5, 0)), 16)
                self.assertEqual(r.wait(make_retry_state(6, 0)), 32)
                self.assertEqual(r.wait(make_retry_state(7, 0)), 64)
                self.assertEqual(r.wait(make_retry_state(8, 0)), 100)
                self.assertEqual(r.wait(make_retry_state(9, 0)), 100)
                self.assertEqual(r.wait(make_retry_state(20, 0)), 100)

    def test_legacy_explicit_wait_type(self):
        if False:
            print('Hello World!')
        Retrying(wait='exponential_sleep')

    def test_wait_func(self):
        if False:
            print('Hello World!')

        def wait_func(retry_state):
            if False:
                return 10
            return retry_state.attempt_number * retry_state.seconds_since_start
        r = Retrying(wait=wait_func)
        self.assertEqual(r.wait(make_retry_state(1, 5)), 5)
        self.assertEqual(r.wait(make_retry_state(2, 11)), 22)
        self.assertEqual(r.wait(make_retry_state(10, 100)), 1000)

    def test_wait_combine(self):
        if False:
            for i in range(10):
                print('nop')
        r = Retrying(wait=tenacity.wait_combine(tenacity.wait_random(0, 3), tenacity.wait_fixed(5)))
        for i in range(1000):
            w = r.wait(make_retry_state(1, 5))
            self.assertLess(w, 8)
            self.assertGreaterEqual(w, 5)

    def test_wait_double_sum(self):
        if False:
            i = 10
            return i + 15
        r = Retrying(wait=tenacity.wait_random(0, 3) + tenacity.wait_fixed(5))
        for i in range(1000):
            w = r.wait(make_retry_state(1, 5))
            self.assertLess(w, 8)
            self.assertGreaterEqual(w, 5)

    def test_wait_triple_sum(self):
        if False:
            while True:
                i = 10
        r = Retrying(wait=tenacity.wait_fixed(1) + tenacity.wait_random(0, 3) + tenacity.wait_fixed(5))
        for i in range(1000):
            w = r.wait(make_retry_state(1, 5))
            self.assertLess(w, 9)
            self.assertGreaterEqual(w, 6)

    def test_wait_arbitrary_sum(self):
        if False:
            i = 10
            return i + 15
        r = Retrying(wait=sum([tenacity.wait_fixed(1), tenacity.wait_random(0, 3), tenacity.wait_fixed(5), tenacity.wait_none()]))
        for _ in range(1000):
            w = r.wait(make_retry_state(1, 5))
            self.assertLess(w, 9)
            self.assertGreaterEqual(w, 6)

    def _assert_range(self, wait, min_, max_):
        if False:
            return 10
        self.assertLess(wait, max_)
        self.assertGreaterEqual(wait, min_)

    def _assert_inclusive_range(self, wait, low, high):
        if False:
            print('Hello World!')
        self.assertLessEqual(wait, high)
        self.assertGreaterEqual(wait, low)

    def _assert_inclusive_epsilon(self, wait, target, epsilon):
        if False:
            return 10
        self.assertLessEqual(wait, target + epsilon)
        self.assertGreaterEqual(wait, target - epsilon)

    def test_wait_chain(self):
        if False:
            while True:
                i = 10
        r = Retrying(wait=tenacity.wait_chain(*[tenacity.wait_fixed(1) for i in range(2)] + [tenacity.wait_fixed(4) for i in range(2)] + [tenacity.wait_fixed(8) for i in range(1)]))
        for i in range(10):
            w = r.wait(make_retry_state(i + 1, 1))
            if i < 2:
                self._assert_range(w, 1, 2)
            elif i < 4:
                self._assert_range(w, 4, 5)
            else:
                self._assert_range(w, 8, 9)

    def test_wait_chain_multiple_invocations(self):
        if False:
            return 10
        sleep_intervals = []
        r = Retrying(sleep=sleep_intervals.append, wait=tenacity.wait_chain(*[tenacity.wait_fixed(i + 1) for i in range(3)]), stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_if_result(lambda x: x == 1))

        @r.wraps
        def always_return_1():
            if False:
                print('Hello World!')
            return 1
        self.assertRaises(tenacity.RetryError, always_return_1)
        self.assertEqual(sleep_intervals, [1.0, 2.0, 3.0, 3.0])
        sleep_intervals[:] = []
        self.assertRaises(tenacity.RetryError, always_return_1)
        self.assertEqual(sleep_intervals, [1.0, 2.0, 3.0, 3.0])
        sleep_intervals[:] = []

    def test_wait_random_exponential(self):
        if False:
            while True:
                i = 10
        fn = tenacity.wait_random_exponential(0.5, 60.0)
        for _ in range(1000):
            self._assert_inclusive_range(fn(make_retry_state(1, 0)), 0, 0.5)
            self._assert_inclusive_range(fn(make_retry_state(2, 0)), 0, 1.0)
            self._assert_inclusive_range(fn(make_retry_state(3, 0)), 0, 2.0)
            self._assert_inclusive_range(fn(make_retry_state(4, 0)), 0, 4.0)
            self._assert_inclusive_range(fn(make_retry_state(5, 0)), 0, 8.0)
            self._assert_inclusive_range(fn(make_retry_state(6, 0)), 0, 16.0)
            self._assert_inclusive_range(fn(make_retry_state(7, 0)), 0, 32.0)
            self._assert_inclusive_range(fn(make_retry_state(8, 0)), 0, 60.0)
            self._assert_inclusive_range(fn(make_retry_state(9, 0)), 0, 60.0)
        fn = tenacity.wait_random_exponential(10, 5)
        for _ in range(1000):
            self._assert_inclusive_range(fn(make_retry_state(1, 0)), 0.0, 5.0)
        fn = tenacity.wait_random_exponential()
        fn(make_retry_state(0, 0))

    def test_wait_random_exponential_statistically(self):
        if False:
            for i in range(10):
                print('nop')
        fn = tenacity.wait_random_exponential(0.5, 60.0)
        attempt = []
        for i in range(10):
            attempt.append([fn(make_retry_state(i, 0)) for _ in range(4000)])

        def mean(lst):
            if False:
                return 10
            return float(sum(lst)) / float(len(lst))
        self._assert_inclusive_epsilon(mean(attempt[1]), 0.25, 0.02)
        self._assert_inclusive_epsilon(mean(attempt[2]), 0.5, 0.04)
        self._assert_inclusive_epsilon(mean(attempt[3]), 1, 0.08)
        self._assert_inclusive_epsilon(mean(attempt[4]), 2, 0.16)
        self._assert_inclusive_epsilon(mean(attempt[5]), 4, 0.32)
        self._assert_inclusive_epsilon(mean(attempt[6]), 8, 0.64)
        self._assert_inclusive_epsilon(mean(attempt[7]), 16, 1.28)
        self._assert_inclusive_epsilon(mean(attempt[8]), 30, 2.56)
        self._assert_inclusive_epsilon(mean(attempt[9]), 30, 2.56)

    def test_wait_exponential_jitter(self):
        if False:
            i = 10
            return i + 15
        fn = tenacity.wait_exponential_jitter(max=60)
        for _ in range(1000):
            self._assert_inclusive_range(fn(make_retry_state(1, 0)), 1, 2)
            self._assert_inclusive_range(fn(make_retry_state(2, 0)), 2, 3)
            self._assert_inclusive_range(fn(make_retry_state(3, 0)), 4, 5)
            self._assert_inclusive_range(fn(make_retry_state(4, 0)), 8, 9)
            self._assert_inclusive_range(fn(make_retry_state(5, 0)), 16, 17)
            self._assert_inclusive_range(fn(make_retry_state(6, 0)), 32, 33)
            self.assertEqual(fn(make_retry_state(7, 0)), 60)
            self.assertEqual(fn(make_retry_state(8, 0)), 60)
            self.assertEqual(fn(make_retry_state(9, 0)), 60)
        fn = tenacity.wait_exponential_jitter(10, 5)
        for _ in range(1000):
            self.assertEqual(fn(make_retry_state(1, 0)), 5)
        fn = tenacity.wait_exponential_jitter()
        fn(make_retry_state(0, 0))

    def test_wait_retry_state_attributes(self):
        if False:
            print('Hello World!')

        class ExtractCallState(Exception):
            pass

        def waitfunc(retry_state):
            if False:
                return 10
            raise ExtractCallState(retry_state)
        retrying = Retrying(wait=waitfunc, retry=tenacity.retry_if_exception_type() | tenacity.retry_if_result(lambda result: result == 123))

        def returnval():
            if False:
                i = 10
                return i + 15
            return 123
        try:
            retrying(returnval)
        except ExtractCallState as err:
            retry_state = err.args[0]
        self.assertIs(retry_state.fn, returnval)
        self.assertEqual(retry_state.args, ())
        self.assertEqual(retry_state.kwargs, {})
        self.assertEqual(retry_state.outcome.result(), 123)
        self.assertEqual(retry_state.attempt_number, 1)
        self.assertGreaterEqual(retry_state.outcome_timestamp, retry_state.start_time)

        def dying():
            if False:
                print('Hello World!')
            raise Exception('Broken')
        try:
            retrying(dying)
        except ExtractCallState as err:
            retry_state = err.args[0]
        self.assertIs(retry_state.fn, dying)
        self.assertEqual(retry_state.args, ())
        self.assertEqual(retry_state.kwargs, {})
        self.assertEqual(str(retry_state.outcome.exception()), 'Broken')
        self.assertEqual(retry_state.attempt_number, 1)
        self.assertGreaterEqual(retry_state.outcome_timestamp, retry_state.start_time)

class TestRetryConditions(unittest.TestCase):

    def test_retry_if_result(self):
        if False:
            print('Hello World!')
        retry = tenacity.retry_if_result(lambda x: x == 1)

        def r(fut):
            if False:
                return 10
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 1, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 2, False)))

    def test_retry_if_not_result(self):
        if False:
            return 10
        retry = tenacity.retry_if_not_result(lambda x: x == 1)

        def r(fut):
            if False:
                while True:
                    i = 10
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 2, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 1, False)))

    def test_retry_any(self):
        if False:
            return 10
        retry = tenacity.retry_any(tenacity.retry_if_result(lambda x: x == 1), tenacity.retry_if_result(lambda x: x == 2))

        def r(fut):
            if False:
                return 10
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 1, False)))
        self.assertTrue(r(tenacity.Future.construct(1, 2, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 3, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 1, True)))

    def test_retry_all(self):
        if False:
            for i in range(10):
                print('nop')
        retry = tenacity.retry_all(tenacity.retry_if_result(lambda x: x == 1), tenacity.retry_if_result(lambda x: isinstance(x, int)))

        def r(fut):
            if False:
                return 10
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 1, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 2, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 3, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 1, True)))

    def test_retry_and(self):
        if False:
            for i in range(10):
                print('nop')
        retry = tenacity.retry_if_result(lambda x: x == 1) & tenacity.retry_if_result(lambda x: isinstance(x, int))

        def r(fut):
            if False:
                print('Hello World!')
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 1, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 2, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 3, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 1, True)))

    def test_retry_or(self):
        if False:
            print('Hello World!')
        retry = tenacity.retry_if_result(lambda x: x == 'foo') | tenacity.retry_if_result(lambda x: isinstance(x, int))

        def r(fut):
            if False:
                return 10
            retry_state = make_retry_state(1, 1.0, last_result=fut)
            return retry(retry_state)
        self.assertTrue(r(tenacity.Future.construct(1, 'foo', False)))
        self.assertFalse(r(tenacity.Future.construct(1, 'foobar', False)))
        self.assertFalse(r(tenacity.Future.construct(1, 2.2, False)))
        self.assertFalse(r(tenacity.Future.construct(1, 42, True)))

    def _raise_try_again(self):
        if False:
            i = 10
            return i + 15
        self._attempts += 1
        if self._attempts < 3:
            raise tenacity.TryAgain

    def test_retry_try_again(self):
        if False:
            for i in range(10):
                print('nop')
        self._attempts = 0
        Retrying(stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_never)(self._raise_try_again)
        self.assertEqual(3, self._attempts)

    def test_retry_try_again_forever(self):
        if False:
            while True:
                i = 10

        def _r():
            if False:
                print('Hello World!')
            raise tenacity.TryAgain
        r = Retrying(stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_never)
        self.assertRaises(tenacity.RetryError, r, _r)
        self.assertEqual(5, r.statistics['attempt_number'])

    def test_retry_try_again_forever_reraise(self):
        if False:
            print('Hello World!')

        def _r():
            if False:
                while True:
                    i = 10
            raise tenacity.TryAgain
        r = Retrying(stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_never, reraise=True)
        self.assertRaises(tenacity.TryAgain, r, _r)
        self.assertEqual(5, r.statistics['attempt_number'])

    def test_retry_if_exception_message_negative_no_inputs(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            tenacity.retry_if_exception_message()

    def test_retry_if_exception_message_negative_too_many_inputs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            tenacity.retry_if_exception_message(message='negative', match='negative')

class NoneReturnUntilAfterCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            for i in range(10):
                print('nop')
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            while True:
                i = 10
        'Return None until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            return None
        return True

class NoIOErrorAfterCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            for i in range(10):
                print('nop')
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            i = 10
            return i + 15
        'Raise an IOError until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            raise OSError("Hi there, I'm an IOError")
        return True

class NoNameErrorAfterCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            print('Hello World!')
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            print('Hello World!')
        'Raise a NameError until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            raise NameError("Hi there, I'm a NameError")
        return True

class NoNameErrorCauseAfterCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            print('Hello World!')
        self.counter = 0
        self.count = count

    def go2(self):
        if False:
            while True:
                i = 10
        raise NameError("Hi there, I'm a NameError")

    def go(self):
        if False:
            while True:
                i = 10
        'Raise an IOError with a NameError as cause until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            try:
                self.go2()
            except NameError as e:
                raise OSError() from e
        return True

class NoIOErrorCauseAfterCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            for i in range(10):
                print('nop')
        self.counter = 0
        self.count = count

    def go2(self):
        if False:
            print('Hello World!')
        raise OSError("Hi there, I'm an IOError")

    def go(self):
        if False:
            while True:
                i = 10
        'Raise a NameError with an IOError as cause until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            try:
                self.go2()
            except OSError as e:
                raise NameError() from e
        return True

class NameErrorUntilCount:
    """Holds counter state for invoking a method several times in a row."""
    derived_message = "Hi there, I'm a NameError"

    def __init__(self, count):
        if False:
            while True:
                i = 10
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True until after count threshold has been crossed.\n\n        Then raise a NameError.\n        '
        if self.counter < self.count:
            self.counter += 1
            return True
        raise NameError(self.derived_message)

class IOErrorUntilCount:
    """Holds counter state for invoking a method several times in a row."""

    def __init__(self, count):
        if False:
            print('Hello World!')
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            while True:
                i = 10
        'Return True until after count threshold has been crossed.\n\n        Then raise an IOError.\n        '
        if self.counter < self.count:
            self.counter += 1
            return True
        raise OSError("Hi there, I'm an IOError")

class CustomError(Exception):
    """This is a custom exception class.

    Note that For Python 2.x, we don't strictly need to extend BaseException,
    however, Python 3.x will complain. While this test suite won't run
    correctly under Python 3.x without extending from the Python exception
    hierarchy, the actual module code is backwards compatible Python 2.x and
    will allow for cases where exception classes don't extend from the
    hierarchy.
    """

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __str__(self):
        if False:
            print('Hello World!')
        return self.value

class NoCustomErrorAfterCount:
    """Holds counter state for invoking a method several times in a row."""
    derived_message = 'This is a Custom exception class'

    def __init__(self, count):
        if False:
            return 10
        self.counter = 0
        self.count = count

    def go(self):
        if False:
            print('Hello World!')
        'Raise a CustomError until after count threshold has been crossed.\n\n        Then return True.\n        '
        if self.counter < self.count:
            self.counter += 1
            raise CustomError(self.derived_message)
        return True

class CapturingHandler(logging.Handler):
    """Captures log records for inspection."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        if False:
            for i in range(10):
                print('nop')
        self.records.append(record)

def current_time_ms():
    if False:
        i = 10
        return i + 15
    return int(round(time.time() * 1000))

@retry(wait=tenacity.wait_fixed(0.05), retry=tenacity.retry_if_result(lambda result: result is None))
def _retryable_test_with_wait(thing):
    if False:
        return 10
    return thing.go()

@retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_result(lambda result: result is None))
def _retryable_test_with_stop(thing):
    if False:
        for i in range(10):
            print('nop')
    return thing.go()

@retry(retry=tenacity.retry_if_exception_cause_type(NameError))
def _retryable_test_with_exception_cause_type(thing):
    if False:
        return 10
    return thing.go()

@retry(retry=tenacity.retry_if_exception_type(IOError))
def _retryable_test_with_exception_type_io(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(retry=tenacity.retry_if_not_exception_type(IOError))
def _retryable_test_if_not_exception_type_io(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(IOError))
def _retryable_test_with_exception_type_io_attempt_limit(thing):
    if False:
        while True:
            i = 10
    return thing.go()

@retry(retry=tenacity.retry_unless_exception_type(NameError))
def _retryable_test_with_unless_exception_type_name(thing):
    if False:
        return 10
    return thing.go()

@retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_unless_exception_type(NameError))
def _retryable_test_with_unless_exception_type_name_attempt_limit(thing):
    if False:
        while True:
            i = 10
    return thing.go()

@retry(retry=tenacity.retry_unless_exception_type())
def _retryable_test_with_unless_exception_type_no_input(thing):
    if False:
        while True:
            i = 10
    return thing.go()

@retry(stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_if_exception_message(message=NoCustomErrorAfterCount.derived_message))
def _retryable_test_if_exception_message_message(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(retry=tenacity.retry_if_not_exception_message(message=NoCustomErrorAfterCount.derived_message))
def _retryable_test_if_not_exception_message_message(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(retry=tenacity.retry_if_exception_message(match=NoCustomErrorAfterCount.derived_message[:3] + '.*'))
def _retryable_test_if_exception_message_match(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(retry=tenacity.retry_if_not_exception_message(match=NoCustomErrorAfterCount.derived_message[:3] + '.*'))
def _retryable_test_if_not_exception_message_match(thing):
    if False:
        for i in range(10):
            print('nop')
    return thing.go()

@retry(retry=tenacity.retry_if_not_exception_message(message=NameErrorUntilCount.derived_message))
def _retryable_test_not_exception_message_delay(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry
def _retryable_default(thing):
    if False:
        print('Hello World!')
    return thing.go()

@retry()
def _retryable_default_f(thing):
    if False:
        return 10
    return thing.go()

@retry(retry=tenacity.retry_if_exception_type(CustomError))
def _retryable_test_with_exception_type_custom(thing):
    if False:
        i = 10
        return i + 15
    return thing.go()

@retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(CustomError))
def _retryable_test_with_exception_type_custom_attempt_limit(thing):
    if False:
        for i in range(10):
            print('nop')
    return thing.go()

class TestDecoratorWrapper(unittest.TestCase):

    def test_with_wait(self):
        if False:
            return 10
        start = current_time_ms()
        result = _retryable_test_with_wait(NoneReturnUntilAfterCount(5))
        t = current_time_ms() - start
        self.assertGreaterEqual(t, 250)
        self.assertTrue(result)

    def test_with_stop_on_return_value(self):
        if False:
            i = 10
            return i + 15
        try:
            _retryable_test_with_stop(NoneReturnUntilAfterCount(5))
            self.fail('Expected RetryError after 3 attempts')
        except RetryError as re:
            self.assertFalse(re.last_attempt.failed)
            self.assertEqual(3, re.last_attempt.attempt_number)
            self.assertTrue(re.last_attempt.result() is None)
            print(re)

    def test_with_stop_on_exception(self):
        if False:
            i = 10
            return i + 15
        try:
            _retryable_test_with_stop(NoIOErrorAfterCount(5))
            self.fail('Expected IOError')
        except OSError as re:
            self.assertTrue(isinstance(re, IOError))
            print(re)

    def test_retry_if_exception_of_type(self):
        if False:
            print('Hello World!')
        self.assertTrue(_retryable_test_with_exception_type_io(NoIOErrorAfterCount(5)))
        try:
            _retryable_test_with_exception_type_io(NoNameErrorAfterCount(5))
            self.fail('Expected NameError')
        except NameError as n:
            self.assertTrue(isinstance(n, NameError))
            print(n)
        self.assertTrue(_retryable_test_with_exception_type_custom(NoCustomErrorAfterCount(5)))
        try:
            _retryable_test_with_exception_type_custom(NoNameErrorAfterCount(5))
            self.fail('Expected NameError')
        except NameError as n:
            self.assertTrue(isinstance(n, NameError))
            print(n)

    def test_retry_except_exception_of_type(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(_retryable_test_if_not_exception_type_io(NoNameErrorAfterCount(5)))
        try:
            _retryable_test_if_not_exception_type_io(NoIOErrorAfterCount(5))
            self.fail('Expected IOError')
        except OSError as err:
            self.assertTrue(isinstance(err, IOError))
            print(err)

    def test_retry_until_exception_of_type_attempt_number(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.assertTrue(_retryable_test_with_unless_exception_type_name(NameErrorUntilCount(5)))
        except NameError as e:
            s = _retryable_test_with_unless_exception_type_name.retry.statistics
            self.assertTrue(s['attempt_number'] == 6)
            print(e)
        else:
            self.fail('Expected NameError')

    def test_retry_until_exception_of_type_no_type(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.assertTrue(_retryable_test_with_unless_exception_type_no_input(NameErrorUntilCount(5)))
        except NameError as e:
            s = _retryable_test_with_unless_exception_type_no_input.retry.statistics
            self.assertTrue(s['attempt_number'] == 6)
            print(e)
        else:
            self.fail('Expected NameError')

    def test_retry_until_exception_of_type_wrong_exception(self):
        if False:
            print('Hello World!')
        try:
            _retryable_test_with_unless_exception_type_name_attempt_limit(IOErrorUntilCount(2))
            self.fail('Expected RetryError')
        except RetryError as e:
            self.assertTrue(isinstance(e, RetryError))
            print(e)

    def test_retry_if_exception_message(self):
        if False:
            i = 10
            return i + 15
        try:
            self.assertTrue(_retryable_test_if_exception_message_message(NoCustomErrorAfterCount(3)))
        except CustomError:
            print(_retryable_test_if_exception_message_message.retry.statistics)
            self.fail("CustomError should've been retried from errormessage")

    def test_retry_if_not_exception_message(self):
        if False:
            i = 10
            return i + 15
        try:
            self.assertTrue(_retryable_test_if_not_exception_message_message(NoCustomErrorAfterCount(2)))
        except CustomError:
            s = _retryable_test_if_not_exception_message_message.retry.statistics
            self.assertTrue(s['attempt_number'] == 1)

    def test_retry_if_not_exception_message_delay(self):
        if False:
            print('Hello World!')
        try:
            self.assertTrue(_retryable_test_not_exception_message_delay(NameErrorUntilCount(3)))
        except NameError:
            s = _retryable_test_not_exception_message_delay.retry.statistics
            print(s['attempt_number'])
            self.assertTrue(s['attempt_number'] == 4)

    def test_retry_if_exception_message_match(self):
        if False:
            i = 10
            return i + 15
        try:
            self.assertTrue(_retryable_test_if_exception_message_match(NoCustomErrorAfterCount(3)))
        except CustomError:
            self.fail("CustomError should've been retried from errormessage")

    def test_retry_if_not_exception_message_match(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.assertTrue(_retryable_test_if_not_exception_message_message(NoCustomErrorAfterCount(2)))
        except CustomError:
            s = _retryable_test_if_not_exception_message_message.retry.statistics
            self.assertTrue(s['attempt_number'] == 1)

    def test_retry_if_exception_cause_type(self):
        if False:
            print('Hello World!')
        self.assertTrue(_retryable_test_with_exception_cause_type(NoNameErrorCauseAfterCount(5)))
        try:
            _retryable_test_with_exception_cause_type(NoIOErrorCauseAfterCount(5))
            self.fail('Expected exception without NameError as cause')
        except NameError:
            pass

    def test_retry_preserves_argument_defaults(self):
        if False:
            while True:
                i = 10

        def function_with_defaults(a=1):
            if False:
                while True:
                    i = 10
            return a

        def function_with_kwdefaults(*, a=1):
            if False:
                return 10
            return a
        retrying = Retrying(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3))
        wrapped_defaults_function = retrying.wraps(function_with_defaults)
        wrapped_kwdefaults_function = retrying.wraps(function_with_kwdefaults)
        self.assertEqual(function_with_defaults.__defaults__, wrapped_defaults_function.__defaults__)
        self.assertEqual(function_with_kwdefaults.__kwdefaults__, wrapped_kwdefaults_function.__kwdefaults__)

    def test_defaults(self):
        if False:
            while True:
                i = 10
        self.assertTrue(_retryable_default(NoNameErrorAfterCount(5)))
        self.assertTrue(_retryable_default_f(NoNameErrorAfterCount(5)))
        self.assertTrue(_retryable_default(NoCustomErrorAfterCount(5)))
        self.assertTrue(_retryable_default_f(NoCustomErrorAfterCount(5)))

    def test_retry_function_object(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that funсtools.wraps doesn't cause problems with callable objects.\n\n        It raises an error upon trying to wrap it in Py2, because __name__\n        attribute is missing. It's fixed in Py3 but was never backported.\n        "

        class Hello:

            def __call__(self):
                if False:
                    while True:
                        i = 10
                return 'Hello'
        retrying = Retrying(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3))
        h = retrying.wraps(Hello())
        self.assertEqual(h(), 'Hello')

class TestRetryWith:

    def test_redefine_wait(self):
        if False:
            print('Hello World!')
        start = current_time_ms()
        result = _retryable_test_with_wait.retry_with(wait=tenacity.wait_fixed(0.1))(NoneReturnUntilAfterCount(5))
        t = current_time_ms() - start
        assert t >= 500
        assert result is True

    def test_redefine_stop(self):
        if False:
            print('Hello World!')
        result = _retryable_test_with_stop.retry_with(stop=tenacity.stop_after_attempt(5))(NoneReturnUntilAfterCount(4))
        assert result is True

    def test_retry_error_cls_should_be_preserved(self):
        if False:
            while True:
                i = 10

        @retry(stop=tenacity.stop_after_attempt(10), retry_error_cls=ValueError)
        def _retryable():
            if False:
                return 10
            raise Exception('raised for test purposes')
        with pytest.raises(Exception) as exc_ctx:
            _retryable.retry_with(stop=tenacity.stop_after_attempt(2))()
        assert exc_ctx.type is ValueError, 'Should remap to specific exception type'

    def test_retry_error_callback_should_be_preserved(self):
        if False:
            print('Hello World!')

        def return_text(retry_state):
            if False:
                print('Hello World!')
            return 'Calling {} keeps raising errors after {} attempts'.format(retry_state.fn.__name__, retry_state.attempt_number)

        @retry(stop=tenacity.stop_after_attempt(10), retry_error_callback=return_text)
        def _retryable():
            if False:
                i = 10
                return i + 15
            raise Exception('raised for test purposes')
        result = _retryable.retry_with(stop=tenacity.stop_after_attempt(5))()
        assert result == 'Calling _retryable keeps raising errors after 5 attempts'

class TestBeforeAfterAttempts(unittest.TestCase):
    _attempt_number = 0

    def test_before_attempts(self):
        if False:
            while True:
                i = 10
        TestBeforeAfterAttempts._attempt_number = 0

        def _before(retry_state):
            if False:
                for i in range(10):
                    print('nop')
            TestBeforeAfterAttempts._attempt_number = retry_state.attempt_number

        @retry(wait=tenacity.wait_fixed(1), stop=tenacity.stop_after_attempt(1), before=_before)
        def _test_before():
            if False:
                while True:
                    i = 10
            pass
        _test_before()
        self.assertTrue(TestBeforeAfterAttempts._attempt_number == 1)

    def test_after_attempts(self):
        if False:
            print('Hello World!')
        TestBeforeAfterAttempts._attempt_number = 0

        def _after(retry_state):
            if False:
                i = 10
                return i + 15
            TestBeforeAfterAttempts._attempt_number = retry_state.attempt_number

        @retry(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(3), after=_after)
        def _test_after():
            if False:
                for i in range(10):
                    print('nop')
            if TestBeforeAfterAttempts._attempt_number < 2:
                raise Exception('testing after_attempts handler')
            else:
                pass
        _test_after()
        self.assertTrue(TestBeforeAfterAttempts._attempt_number == 2)

    def test_before_sleep(self):
        if False:
            i = 10
            return i + 15

        def _before_sleep(retry_state):
            if False:
                i = 10
                return i + 15
            self.assertGreater(retry_state.next_action.sleep, 0)
            _before_sleep.attempt_number = retry_state.attempt_number

        @retry(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)
        def _test_before_sleep():
            if False:
                print('Hello World!')
            if _before_sleep.attempt_number < 2:
                raise Exception('testing before_sleep_attempts handler')
        _test_before_sleep()
        self.assertEqual(_before_sleep.attempt_number, 2)

    def _before_sleep_log_raises(self, get_call_fn):
        if False:
            return 10
        thing = NoIOErrorAfterCount(2)
        logger = logging.getLogger(self.id())
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            _before_sleep = tenacity.before_sleep_log(logger, logging.INFO)
            retrying = Retrying(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)
            get_call_fn(retrying)(thing.go)
        finally:
            logger.removeHandler(handler)
        etalon_re = "^Retrying .* in 0\\.01 seconds as it raised (IO|OS)Error: Hi there, I'm an IOError\\.$"
        self.assertEqual(len(handler.records), 2)
        fmt = logging.Formatter().format
        self.assertRegex(fmt(handler.records[0]), etalon_re)
        self.assertRegex(fmt(handler.records[1]), etalon_re)

    def test_before_sleep_log_raises(self):
        if False:
            print('Hello World!')
        self._before_sleep_log_raises(lambda x: x)

    def test_before_sleep_log_raises_with_exc_info(self):
        if False:
            print('Hello World!')
        thing = NoIOErrorAfterCount(2)
        logger = logging.getLogger(self.id())
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            _before_sleep = tenacity.before_sleep_log(logger, logging.INFO, exc_info=True)
            retrying = Retrying(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)
            retrying(thing.go)
        finally:
            logger.removeHandler(handler)
        etalon_re = re.compile("^Retrying .* in 0\\.01 seconds as it raised (IO|OS)Error: Hi there, I'm an IOError\\.{0}Traceback \\(most recent call last\\):{0}.*$".format('\n'), flags=re.MULTILINE)
        self.assertEqual(len(handler.records), 2)
        fmt = logging.Formatter().format
        self.assertRegex(fmt(handler.records[0]), etalon_re)
        self.assertRegex(fmt(handler.records[1]), etalon_re)

    def test_before_sleep_log_returns(self, exc_info=False):
        if False:
            i = 10
            return i + 15
        thing = NoneReturnUntilAfterCount(2)
        logger = logging.getLogger(self.id())
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            _before_sleep = tenacity.before_sleep_log(logger, logging.INFO, exc_info=exc_info)
            _retry = tenacity.retry_if_result(lambda result: result is None)
            retrying = Retrying(wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3), retry=_retry, before_sleep=_before_sleep)
            retrying(thing.go)
        finally:
            logger.removeHandler(handler)
        etalon_re = '^Retrying .* in 0\\.01 seconds as it returned None\\.$'
        self.assertEqual(len(handler.records), 2)
        fmt = logging.Formatter().format
        self.assertRegex(fmt(handler.records[0]), etalon_re)
        self.assertRegex(fmt(handler.records[1]), etalon_re)

    def test_before_sleep_log_returns_with_exc_info(self):
        if False:
            i = 10
            return i + 15
        self.test_before_sleep_log_returns(exc_info=True)

class TestReraiseExceptions(unittest.TestCase):

    def test_reraise_by_default(self):
        if False:
            for i in range(10):
                print('nop')
        calls = []

        @retry(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(2), reraise=True)
        def _reraised_by_default():
            if False:
                return 10
            calls.append('x')
            raise KeyError('Bad key')
        self.assertRaises(KeyError, _reraised_by_default)
        self.assertEqual(2, len(calls))

    def test_reraise_from_retry_error(self):
        if False:
            for i in range(10):
                print('nop')
        calls = []

        @retry(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(2))
        def _raise_key_error():
            if False:
                return 10
            calls.append('x')
            raise KeyError('Bad key')

        def _reraised_key_error():
            if False:
                i = 10
                return i + 15
            try:
                _raise_key_error()
            except tenacity.RetryError as retry_err:
                retry_err.reraise()
        self.assertRaises(KeyError, _reraised_key_error)
        self.assertEqual(2, len(calls))

    def test_reraise_timeout_from_retry_error(self):
        if False:
            print('Hello World!')
        calls = []

        @retry(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(2), retry=lambda retry_state: True)
        def _mock_fn():
            if False:
                return 10
            calls.append('x')

        def _reraised_mock_fn():
            if False:
                for i in range(10):
                    print('nop')
            try:
                _mock_fn()
            except tenacity.RetryError as retry_err:
                retry_err.reraise()
        self.assertRaises(tenacity.RetryError, _reraised_mock_fn)
        self.assertEqual(2, len(calls))

    def test_reraise_no_exception(self):
        if False:
            for i in range(10):
                print('nop')
        calls = []

        @retry(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(2), retry=lambda retry_state: True, reraise=True)
        def _mock_fn():
            if False:
                while True:
                    i = 10
            calls.append('x')
        self.assertRaises(tenacity.RetryError, _mock_fn)
        self.assertEqual(2, len(calls))

class TestStatistics(unittest.TestCase):

    def test_stats(self):
        if False:
            return 10

        @retry()
        def _foobar():
            if False:
                print('Hello World!')
            return 42
        self.assertEqual({}, _foobar.retry.statistics)
        _foobar()
        self.assertEqual(1, _foobar.retry.statistics['attempt_number'])

    def test_stats_failing(self):
        if False:
            i = 10
            return i + 15

        @retry(stop=tenacity.stop_after_attempt(2))
        def _foobar():
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError(42)
        self.assertEqual({}, _foobar.retry.statistics)
        try:
            _foobar()
        except Exception:
            pass
        self.assertEqual(2, _foobar.retry.statistics['attempt_number'])

class TestRetryErrorCallback(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._attempt_number = 0
        self._callback_called = False

    def _callback(self, fut):
        if False:
            print('Hello World!')
        self._callback_called = True
        return fut

    def test_retry_error_callback(self):
        if False:
            return 10
        num_attempts = 3

        def retry_error_callback(retry_state):
            if False:
                return 10
            retry_error_callback.called_times += 1
            return retry_state.outcome
        retry_error_callback.called_times = 0

        @retry(stop=tenacity.stop_after_attempt(num_attempts), retry_error_callback=retry_error_callback)
        def _foobar():
            if False:
                print('Hello World!')
            self._attempt_number += 1
            raise Exception('This exception should not be raised')
        result = _foobar()
        self.assertEqual(retry_error_callback.called_times, 1)
        self.assertEqual(num_attempts, self._attempt_number)
        self.assertIsInstance(result, tenacity.Future)

class TestContextManager(unittest.TestCase):

    def test_context_manager_retry_one(self):
        if False:
            while True:
                i = 10
        from tenacity import Retrying
        raise_ = True
        for attempt in Retrying():
            with attempt:
                if raise_:
                    raise_ = False
                    raise Exception('Retry it!')

    def test_context_manager_on_error(self):
        if False:
            return 10
        from tenacity import Retrying

        class CustomError(Exception):
            pass
        retry = Retrying(retry=tenacity.retry_if_exception_type(IOError))

        def test():
            if False:
                for i in range(10):
                    print('nop')
            for attempt in retry:
                with attempt:
                    raise CustomError("Don't retry!")
        self.assertRaises(CustomError, test)

    def test_context_manager_retry_error(self):
        if False:
            return 10
        from tenacity import Retrying
        retry = Retrying(stop=tenacity.stop_after_attempt(2))

        def test():
            if False:
                return 10
            for attempt in retry:
                with attempt:
                    raise Exception('Retry it!')
        self.assertRaises(RetryError, test)

    def test_context_manager_reraise(self):
        if False:
            return 10
        from tenacity import Retrying

        class CustomError(Exception):
            pass
        retry = Retrying(reraise=True, stop=tenacity.stop_after_attempt(2))

        def test():
            if False:
                while True:
                    i = 10
            for attempt in retry:
                with attempt:
                    raise CustomError("Don't retry!")
        self.assertRaises(CustomError, test)

class TestInvokeAsCallable:
    """Test direct invocation of Retrying as a callable."""

    @staticmethod
    def invoke(retry, f):
        if False:
            while True:
                i = 10
        '\n        Invoke Retrying logic.\n\n        Wrapper allows testing different call mechanisms in test sub-classes.\n        '
        return retry(f)

    def test_retry_one(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            f.calls.append(len(f.calls) + 1)
            if len(f.calls) <= 1:
                raise Exception('Retry it!')
            return 42
        f.calls = []
        retry = Retrying()
        assert self.invoke(retry, f) == 42
        assert f.calls == [1, 2]

    def test_on_error(self):
        if False:
            return 10

        class CustomError(Exception):
            pass

        def f():
            if False:
                i = 10
                return i + 15
            f.calls.append(len(f.calls) + 1)
            if len(f.calls) <= 1:
                raise CustomError("Don't retry!")
            return 42
        f.calls = []
        retry = Retrying(retry=tenacity.retry_if_exception_type(IOError))
        with pytest.raises(CustomError):
            self.invoke(retry, f)
        assert f.calls == [1]

    def test_retry_error(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            f.calls.append(len(f.calls) + 1)
            raise Exception('Retry it!')
        f.calls = []
        retry = Retrying(stop=tenacity.stop_after_attempt(2))
        with pytest.raises(RetryError):
            self.invoke(retry, f)
        assert f.calls == [1, 2]

    def test_reraise(self):
        if False:
            while True:
                i = 10

        class CustomError(Exception):
            pass

        def f():
            if False:
                print('Hello World!')
            f.calls.append(len(f.calls) + 1)
            raise CustomError('Retry it!')
        f.calls = []
        retry = Retrying(reraise=True, stop=tenacity.stop_after_attempt(2))
        with pytest.raises(CustomError):
            self.invoke(retry, f)
        assert f.calls == [1, 2]

class TestRetryException(unittest.TestCase):

    def test_retry_error_is_pickleable(self):
        if False:
            i = 10
            return i + 15
        import pickle
        expected = RetryError(last_attempt=123)
        pickled = pickle.dumps(expected)
        actual = pickle.loads(pickled)
        self.assertEqual(expected.last_attempt, actual.last_attempt)

class TestRetryTyping(unittest.TestCase):

    @pytest.mark.skipif(sys.version_info < (3, 0), reason='typeguard not supported for python 2')
    def test_retry_type_annotations(self):
        if False:
            return 10
        'The decorator should maintain types of decorated functions.'
        if sys.version_info < (3, 0):
            return
        from typeguard import check_type

        def num_to_str(number):
            if False:
                return 10
            return str(number)
        with_raw = retry(num_to_str)
        with_raw_result = with_raw(1)
        with_constructor = retry()(num_to_str)
        with_constructor_result = with_raw(1)
        check_type(with_raw, typing.Callable[[int], str])
        check_type(with_raw_result, str)
        check_type(with_constructor, typing.Callable[[int], str])
        check_type(with_constructor_result, str)

@contextmanager
def reports_deprecation_warning():
    if False:
        return 10
    __tracebackhide__ = True
    oldfilters = copy(warnings.filters)
    warnings.simplefilter('always')
    try:
        with pytest.warns(DeprecationWarning):
            yield
    finally:
        warnings.filters = oldfilters

class TestMockingSleep:
    RETRY_ARGS = dict(wait=tenacity.wait_fixed(0.1), stop=tenacity.stop_after_attempt(5))

    def _fail(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @retry(**RETRY_ARGS)
    def _decorated_fail(self):
        if False:
            while True:
                i = 10
        self._fail()

    @pytest.fixture()
    def mock_sleep(self, monkeypatch):
        if False:
            while True:
                i = 10

        class MockSleep:
            call_count = 0

            def __call__(self, seconds):
                if False:
                    print('Hello World!')
                self.call_count += 1
        sleep = MockSleep()
        monkeypatch.setattr(tenacity.nap.time, 'sleep', sleep)
        yield sleep

    def test_decorated(self, mock_sleep):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(RetryError):
            self._decorated_fail()
        assert mock_sleep.call_count == 4

    def test_decorated_retry_with(self, mock_sleep):
        if False:
            return 10
        fail_faster = self._decorated_fail.retry_with(stop=tenacity.stop_after_attempt(2))
        with pytest.raises(RetryError):
            fail_faster()
        assert mock_sleep.call_count == 1
if __name__ == '__main__':
    unittest.main()