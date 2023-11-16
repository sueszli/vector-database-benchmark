"""
Tests for SelectConnection _Timer and _Timeout classes

"""
import math
import time
import unittest
from unittest import mock
import pika.compat
from pika.adapters import select_connection

def test_now():
    if False:
        while True:
            i = 10
    return math.ceil(pika.compat.time_now())

class ChildTimeout(select_connection._Timeout):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ChildTimeout, self).__init__(*args, **kwargs)
        self.extra = 'e'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, ChildTimeout):
            return self.extra == other.extra and super(ChildTimeout, self).__eq__(other)
        return NotImplemented

class TimeoutClassTests(unittest.TestCase):
    """Test select_connection._Timeout class"""

    def test_properties(self):
        if False:
            while True:
                i = 10
        now = test_now()
        cb = lambda : None
        timeout = select_connection._Timeout(now + 5.3, cb)
        self.assertIs(timeout.callback, cb)
        self.assertEqual(timeout.deadline, now + 5.3)

    def test_non_negative_deadline(self):
        if False:
            i = 10
            return i + 15
        select_connection._Timeout(0, lambda : None)
        select_connection._Timeout(5, lambda : None)
        with self.assertRaises(ValueError) as cm:
            select_connection._Timeout(-1, lambda : None)
        self.assertIn('deadline must be non-negative epoch number', cm.exception.args[0])

    def test_non_callable_callback_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError) as cm:
            select_connection._Timeout(5, None)
        self.assertIn('callback must be a callable, but got', cm.exception.args[0])
        with self.assertRaises(TypeError) as cm:
            select_connection._Timeout(5, dict())
        self.assertIn('callback must be a callable, but got', cm.exception.args[0])

    def test_eq(self):
        if False:
            return 10
        self.assertEqual(select_connection._Timeout(5, lambda : None), select_connection._Timeout(5, lambda : 5))
        self.assertEqual(select_connection._Timeout(5, lambda : 5), select_connection._Timeout(5, lambda : None))
        self.assertEqual(select_connection._Timeout(5, lambda : None), ChildTimeout(5, lambda : 5))
        self.assertEqual(ChildTimeout(5, lambda : 5), select_connection._Timeout(5, lambda : None))

        class Foreign(object):

            def __eq__(self, other):
                if False:
                    return 10
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) == Foreign(), 'foobar')
        self.assertEqual(Foreign() == select_connection._Timeout(5, lambda : None), 'foobar')

    def test_ne(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(select_connection._Timeout(5, lambda : None), select_connection._Timeout(10, lambda : None))
        self.assertNotEqual(select_connection._Timeout(10, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertNotEqual(select_connection._Timeout(5, lambda : None), ChildTimeout(10, lambda : None))
        self.assertNotEqual(ChildTimeout(10, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertNotEqual(select_connection._Timeout(5, lambda : None), dict(deadline=5, callback=lambda : None))
        self.assertNotEqual(dict(deadline=5, callback=lambda : None), select_connection._Timeout(5, lambda : None))

        class Foreign(object):

            def __ne__(self, other):
                if False:
                    i = 10
                    return i + 15
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) != Foreign(), 'foobar')
        self.assertEqual(Foreign() != select_connection._Timeout(5, lambda : None), 'foobar')

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertLess(select_connection._Timeout(5, lambda : None), select_connection._Timeout(10, lambda : None))
        self.assertLess(select_connection._Timeout(5, lambda : None), ChildTimeout(10, lambda : None))

        class Foreign(object):

            def __gt__(self, other):
                if False:
                    while True:
                        i = 10
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) < Foreign(), 'foobar')
        self.assertFalse(select_connection._Timeout(5, lambda : None) < select_connection._Timeout(5, lambda : None))
        self.assertFalse(select_connection._Timeout(5, lambda : None) < select_connection._Timeout(1, lambda : None))

    def test_gt(self):
        if False:
            i = 10
            return i + 15
        self.assertGreater(select_connection._Timeout(10, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertGreater(select_connection._Timeout(10, lambda : None), ChildTimeout(5, lambda : None))

        class Foreign(object):

            def __lt__(self, other):
                if False:
                    return 10
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) > Foreign(), 'foobar')
        self.assertFalse(select_connection._Timeout(5, lambda : None) > select_connection._Timeout(5, lambda : None))
        self.assertFalse(select_connection._Timeout(1, lambda : None) > select_connection._Timeout(5, lambda : None))

    def test_le(self):
        if False:
            while True:
                i = 10
        self.assertLessEqual(select_connection._Timeout(5, lambda : None), select_connection._Timeout(10, lambda : None))
        self.assertLessEqual(select_connection._Timeout(5, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertLessEqual(select_connection._Timeout(5, lambda : None), ChildTimeout(10, lambda : None))
        self.assertLessEqual(select_connection._Timeout(5, lambda : None), ChildTimeout(5, lambda : None))

        class Foreign(object):

            def __ge__(self, other):
                if False:
                    while True:
                        i = 10
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) <= Foreign(), 'foobar')
        self.assertFalse(select_connection._Timeout(5, lambda : None) <= select_connection._Timeout(1, lambda : None))

    def test_ge(self):
        if False:
            while True:
                i = 10
        self.assertGreaterEqual(select_connection._Timeout(10, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertGreaterEqual(select_connection._Timeout(5, lambda : None), select_connection._Timeout(5, lambda : None))
        self.assertGreaterEqual(select_connection._Timeout(10, lambda : None), ChildTimeout(5, lambda : None))
        self.assertGreaterEqual(select_connection._Timeout(5, lambda : None), ChildTimeout(5, lambda : None))

        class Foreign(object):

            def __le__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return 'foobar'
        self.assertEqual(select_connection._Timeout(5, lambda : None) >= Foreign(), 'foobar')
        self.assertFalse(select_connection._Timeout(1, lambda : None) >= select_connection._Timeout(5, lambda : None))

class TimerClassTests(unittest.TestCase):
    """Test select_connection._Timer class"""

    def test_close_empty(self):
        if False:
            for i in range(10):
                print('nop')
        timer = select_connection._Timer()
        timer.close()
        self.assertIsNone(timer._timeout_heap)

    def test_close_non_empty(self):
        if False:
            for i in range(10):
                print('nop')
        timer = select_connection._Timer()
        t1 = timer.call_later(10, lambda : 10)
        t2 = timer.call_later(20, lambda : 20)
        timer.close()
        self.assertIsNone(timer._timeout_heap)
        self.assertIsNone(t1.callback)
        self.assertIsNone(t2.callback)

    def test_no_timeouts_remaining_interval_is_none(self):
        if False:
            print('Hello World!')
        timer = select_connection._Timer()
        self.assertIsNone(timer.get_remaining_interval())

    def test_call_later_non_negative_delay_check(self):
        if False:
            for i in range(10):
                print('nop')
        now = test_now()
        with mock.patch('pika.compat.time_now', return_value=now):
            timer = select_connection._Timer()
            timer.call_later(0, lambda : None)
            self.assertEqual(timer._timeout_heap[0].deadline, now)
            self.assertEqual(timer.get_remaining_interval(), 0)
        with mock.patch('pika.compat.time_now', return_value=now):
            timer = select_connection._Timer()
            timer.call_later(0.5, lambda : None)
            self.assertEqual(timer._timeout_heap[0].deadline, now + 0.5)
            self.assertEqual(timer.get_remaining_interval(), 0.5)
        timer = select_connection._Timer()
        with self.assertRaises(ValueError) as cm:
            timer.call_later(-5, lambda : None)
        self.assertIn('call_later: delay must be non-negative, but got', cm.exception.args[0])

    def test_call_later_single_timer_expires(self):
        if False:
            i = 10
            return i + 15
        now = test_now()
        with mock.patch('pika.compat.time_now', return_value=now):
            bucket = []
            timer = select_connection._Timer()
            timer.call_later(5, lambda : bucket.append(1))
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertEqual(timer.get_remaining_interval(), 5)
        with mock.patch('pika.compat.time_now', return_value=now + 5):
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [1])
            self.assertEqual(len(timer._timeout_heap), 0)
            self.assertIsNone(timer.get_remaining_interval())

    def test_call_later_multiple_timers(self):
        if False:
            for i in range(10):
                print('nop')
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch('pika.compat.time_now', return_value=now):
            timer.call_later(5, lambda : bucket.append(1))
            timer.call_later(5, lambda : bucket.append(2))
            timer.call_later(10, lambda : bucket.append(3))
            self.assertEqual(timer.get_remaining_interval(), 5)
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertEqual(timer.get_remaining_interval(), 5)
        with mock.patch('pika.compat.time_now', return_value=now + 6):
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [1, 2])
            self.assertEqual(len(timer._timeout_heap), 1)
            self.assertEqual(timer.get_remaining_interval(), 4)
        with mock.patch('pika.compat.time_now', return_value=now + 10):
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [1, 2, 3])
            self.assertEqual(len(timer._timeout_heap), 0)
            self.assertIsNone(timer.get_remaining_interval())

    def test_add_and_remove_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch('pika.compat.time_now', return_value=now):
            timer.call_later(10, lambda : bucket.append(3))
            t2 = timer.call_later(6, lambda : bucket.append(2))
            t1 = timer.call_later(5, lambda : bucket.append(1))
            self.assertEqual(timer.get_remaining_interval(), 5)
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertEqual(timer.get_remaining_interval(), 5)
            timer.remove_timeout(t1)
            self.assertIsNone(t1.callback)
            self.assertEqual(timer._num_cancellations, 1)
            timer.remove_timeout(t2)
            self.assertIsNone(t2.callback)
            self.assertEqual(timer._num_cancellations, 2)
            self.assertEqual(timer.get_remaining_interval(), 5)
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertEqual(timer._num_cancellations, 2)
            self.assertEqual(timer.get_remaining_interval(), 5)
            self.assertEqual(len(timer._timeout_heap), 3)
        with mock.patch('pika.compat.time_now', return_value=now + 6):
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertEqual(timer._num_cancellations, 0)
            self.assertEqual(len(timer._timeout_heap), 1)
            self.assertEqual(timer.get_remaining_interval(), 4)
        with mock.patch('pika.compat.time_now', return_value=now + 10):
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [3])
            self.assertEqual(len(timer._timeout_heap), 0)
            self.assertIsNone(timer.get_remaining_interval())

    def test_gc_of_unexpired_timeouts(self):
        if False:
            i = 10
            return i + 15
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch.multiple(select_connection._Timer, _GC_CANCELLATION_THRESHOLD=1):
            with mock.patch('pika.compat.time_now', return_value=now):
                t3 = timer.call_later(10, lambda : bucket.append(3))
                t2 = timer.call_later(6, lambda : bucket.append(2))
                t1 = timer.call_later(5, lambda : bucket.append(1))
                timer.remove_timeout(t1)
                self.assertEqual(timer._num_cancellations, 1)
                timer.process_timeouts()
                self.assertEqual(timer._num_cancellations, 1)
                self.assertEqual(bucket, [])
                self.assertEqual(len(timer._timeout_heap), 3)
                self.assertEqual(timer.get_remaining_interval(), 5)
                timer.remove_timeout(t3)
                self.assertEqual(timer._num_cancellations, 2)
                timer.process_timeouts()
                self.assertEqual(bucket, [])
                self.assertEqual(len(timer._timeout_heap), 1)
                self.assertIs(t2, timer._timeout_heap[0])
                self.assertEqual(timer.get_remaining_interval(), 6)
                self.assertEqual(timer._num_cancellations, 0)

    def test_add_timeout_from_another_timeout(self):
        if False:
            print('Hello World!')
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch('pika.compat.time_now', return_value=now):
            t1 = timer.call_later(5, lambda : bucket.append(timer.call_later(0, lambda : bucket.append(2))))
        with mock.patch('pika.compat.time_now', return_value=now + 10):
            timer.process_timeouts()
            t2 = bucket.pop()
            self.assertIsInstance(t2, select_connection._Timeout)
            self.assertIsNot(t2, t1)
            self.assertEqual(bucket, [])
            self.assertEqual(len(timer._timeout_heap), 1)
            self.assertIs(t2, timer._timeout_heap[0])
            self.assertEqual(timer.get_remaining_interval(), 0)
            timer.process_timeouts()
            self.assertEqual(bucket, [2])
            self.assertEqual(timer.get_remaining_interval(), None)

    def test_cancel_unexpired_timeout_from_another_timeout(self):
        if False:
            while True:
                i = 10
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch('pika.compat.time_now', return_value=now):
            t2 = timer.call_later(10, lambda : bucket.append(2))
            t1 = timer.call_later(5, lambda : timer.remove_timeout(t2))
            self.assertIs(t1, timer._timeout_heap[0])
        with mock.patch('pika.compat.time_now', return_value=now + 6):
            timer.process_timeouts()
            self.assertIsNone(t2.callback)
            self.assertEqual(timer.get_remaining_interval(), 4)
            self.assertIs(t2, timer._timeout_heap[0])
            self.assertEqual(timer._num_cancellations, 1)
        with mock.patch('pika.compat.time_now', return_value=now + 10):
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertIsNone(timer.get_remaining_interval())
            self.assertEqual(len(timer._timeout_heap), 0)
            self.assertEqual(timer._num_cancellations, 0)

    def test_cancel_expired_timeout_from_another_timeout(self):
        if False:
            return 10
        now = test_now()
        bucket = []
        timer = select_connection._Timer()
        with mock.patch('pika.compat.time_now', return_value=now):
            t2 = timer.call_later(10, lambda : bucket.append(2))
            t1 = timer.call_later(5, lambda : (self.assertEqual(timer._num_cancellations, 0), timer.remove_timeout(t2)))
            self.assertIs(t1, timer._timeout_heap[0])
        with mock.patch('pika.compat.time_now', return_value=now + 10):
            timer.process_timeouts()
            self.assertEqual(bucket, [])
            self.assertIsNone(t2.callback)
            self.assertIsNone(timer.get_remaining_interval())
            self.assertEqual(len(timer._timeout_heap), 0)
            self.assertEqual(timer._num_cancellations, 0)