"""Counted lock class"""
from __future__ import absolute_import
from bzrlib import errors

class CountedLock(object):
    """Decorator around a lock that makes it reentrant.

    This can be used with any object that provides a basic Lock interface,
    including LockDirs and OS file locks.

    :ivar _token: While a write lock is held, this is the token 
        for it.
    """

    def __init__(self, real_lock):
        if False:
            for i in range(10):
                print('nop')
        self._real_lock = real_lock
        self._lock_mode = None
        self._lock_count = 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r)' % (self.__class__.__name__, self._real_lock)

    def break_lock(self):
        if False:
            for i in range(10):
                print('nop')
        self._real_lock.break_lock()
        self._lock_mode = None
        self._lock_count = 0

    def get_physical_lock_status(self):
        if False:
            i = 10
            return i + 15
        'Return physical lock status.\n\n        Returns true if a lock is held on the transport. If no lock is held, or\n        the underlying locking mechanism does not support querying lock\n        status, false is returned.\n        '
        try:
            return self._real_lock.peek() is not None
        except NotImplementedError:
            return False

    def is_locked(self):
        if False:
            return 10
        return self._lock_mode is not None

    def lock_read(self):
        if False:
            return 10
        'Acquire the lock in read mode.\n\n        If the lock is already held in either read or write mode this\n        increments the count and succeeds.  If the lock is not already held,\n        it is taken in read mode.\n        '
        if self._lock_mode:
            self._lock_count += 1
        else:
            self._real_lock.lock_read()
            self._lock_count = 1
            self._lock_mode = 'r'

    def lock_write(self, token=None):
        if False:
            for i in range(10):
                print('nop')
        'Acquire the lock in write mode.\n\n        If the lock was originally acquired in read mode this will fail.\n\n        :param token: If given and the lock is already held, \n            then validate that we already hold the real\n            lock with this token.\n\n        :returns: The token from the underlying lock.\n        '
        if self._lock_count == 0:
            self._token = self._real_lock.lock_write(token=token)
            self._lock_mode = 'w'
            self._lock_count += 1
            return self._token
        elif self._lock_mode != 'w':
            raise errors.ReadOnlyError(self)
        else:
            self._real_lock.validate_token(token)
            self._lock_count += 1
            return self._token

    def unlock(self):
        if False:
            i = 10
            return i + 15
        if self._lock_count == 0:
            raise errors.LockNotHeld(self)
        elif self._lock_count == 1:
            self._lock_mode = None
            self._lock_count -= 1
            self._real_lock.unlock()
        else:
            self._lock_count -= 1