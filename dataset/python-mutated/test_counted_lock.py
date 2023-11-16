"""Tests for bzrlib.counted_lock"""
from bzrlib.counted_lock import CountedLock
from bzrlib.errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from bzrlib.tests import TestCase

class DummyLock(object):
    """Lock that just records what's been done to it."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._calls = []
        self._lock_mode = None

    def is_locked(self):
        if False:
            for i in range(10):
                print('nop')
        return self._lock_mode is not None

    def lock_read(self):
        if False:
            i = 10
            return i + 15
        self._assert_not_locked()
        self._lock_mode = 'r'
        self._calls.append('lock_read')

    def lock_write(self, token=None):
        if False:
            print('Hello World!')
        if token is not None:
            if token == 'token':
                return 'token'
            else:
                raise TokenMismatch()
        self._assert_not_locked()
        self._lock_mode = 'w'
        self._calls.append('lock_write')
        return 'token'

    def unlock(self):
        if False:
            i = 10
            return i + 15
        self._assert_locked()
        self._lock_mode = None
        self._calls.append('unlock')

    def break_lock(self):
        if False:
            for i in range(10):
                print('nop')
        self._lock_mode = None
        self._calls.append('break')

    def _assert_locked(self):
        if False:
            while True:
                i = 10
        if not self._lock_mode:
            raise LockError('%s is not locked' % (self,))

    def _assert_not_locked(self):
        if False:
            i = 10
            return i + 15
        if self._lock_mode:
            raise LockError('%s is already locked in mode %r' % (self, self._lock_mode))

    def validate_token(self, token):
        if False:
            i = 10
            return i + 15
        if token == 'token':
            return 'token'
        elif token is None:
            return
        else:
            raise TokenMismatch(token, 'token')

class TestDummyLock(TestCase):

    def test_lock_initially_not_held(self):
        if False:
            for i in range(10):
                print('nop')
        l = DummyLock()
        self.assertFalse(l.is_locked())

    def test_lock_not_reentrant(self):
        if False:
            while True:
                i = 10
        l = DummyLock()
        l.lock_read()
        self.assertRaises(LockError, l.lock_read)

    def test_detect_underlock(self):
        if False:
            return 10
        l = DummyLock()
        self.assertRaises(LockError, l.unlock)

    def test_basic_locking(self):
        if False:
            for i in range(10):
                print('nop')
        real_lock = DummyLock()
        self.assertFalse(real_lock.is_locked())
        real_lock.lock_read()
        self.assertTrue(real_lock.is_locked())
        real_lock.unlock()
        self.assertFalse(real_lock.is_locked())
        result = real_lock.lock_write()
        self.assertEqual('token', result)
        self.assertTrue(real_lock.is_locked())
        real_lock.unlock()
        self.assertFalse(real_lock.is_locked())
        self.assertEqual(['lock_read', 'unlock', 'lock_write', 'unlock'], real_lock._calls)

    def test_break_lock(self):
        if False:
            print('Hello World!')
        l = DummyLock()
        l.lock_write()
        l.break_lock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_write', 'break'], l._calls)

class TestCountedLock(TestCase):

    def test_read_lock(self):
        if False:
            for i in range(10):
                print('nop')
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertFalse(l.is_locked())
        l.lock_read()
        l.lock_read()
        self.assertTrue(l.is_locked())
        l.unlock()
        self.assertTrue(l.is_locked())
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_read', 'unlock'], real_lock._calls)

    def test_unlock_not_locked(self):
        if False:
            print('Hello World!')
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertRaises(LockNotHeld, l.unlock)

    def test_read_lock_while_write_locked(self):
        if False:
            for i in range(10):
                print('nop')
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_write()
        l.lock_read()
        self.assertEqual('token', l.lock_write())
        l.unlock()
        l.unlock()
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_write', 'unlock'], real_lock._calls)

    def test_write_lock_while_read_locked(self):
        if False:
            for i in range(10):
                print('nop')
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_read()
        self.assertRaises(ReadOnlyError, l.lock_write)
        self.assertRaises(ReadOnlyError, l.lock_write)
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_read', 'unlock'], real_lock._calls)

    def test_write_lock_reentrant(self):
        if False:
            i = 10
            return i + 15
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertEqual('token', l.lock_write())
        self.assertEqual('token', l.lock_write())
        l.unlock()
        l.unlock()

    def test_reenter_with_token(self):
        if False:
            return 10
        real_lock = DummyLock()
        l1 = CountedLock(real_lock)
        l2 = CountedLock(real_lock)
        token = l1.lock_write()
        self.assertEqual('token', token)
        del l1
        self.assertTrue(real_lock.is_locked())
        self.assertFalse(l2.is_locked())
        self.assertEqual(token, l2.lock_write(token=token))
        self.assertTrue(l2.is_locked())
        self.assertTrue(real_lock.is_locked())
        l2.unlock()
        self.assertFalse(l2.is_locked())
        self.assertFalse(real_lock.is_locked())

    def test_break_lock(self):
        if False:
            i = 10
            return i + 15
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_write()
        l.lock_write()
        self.assertTrue(real_lock.is_locked())
        l.break_lock()
        self.assertFalse(l.is_locked())
        self.assertFalse(real_lock.is_locked())