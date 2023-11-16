from __future__ import absolute_import
import time
import os
from . import LockBase, LockFailed, NotLocked, NotMyLock, LockTimeout, AlreadyLocked

class LinkLockFile(LockBase):
    """Lock access to a file using atomic property of link(2).

    >>> lock = LinkLockFile('somefile')
    >>> lock = LinkLockFile('somefile', threaded=False)
    """

    def acquire(self, timeout=None):
        if False:
            i = 10
            return i + 15
        try:
            open(self.unique_name, 'wb').close()
        except IOError:
            raise LockFailed('failed to create %s' % self.unique_name)
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout
        while True:
            try:
                os.link(self.unique_name, self.lock_file)
            except OSError:
                nlinks = os.stat(self.unique_name).st_nlink
                if nlinks == 2:
                    return
                else:
                    if timeout is not None and time.time() > end_time:
                        os.unlink(self.unique_name)
                        if timeout > 0:
                            raise LockTimeout('Timeout waiting to acquire lock for %s' % self.path)
                        else:
                            raise AlreadyLocked('%s is already locked' % self.path)
                    time.sleep(timeout is not None and timeout / 10 or 0.1)
            else:
                return

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_locked():
            raise NotLocked('%s is not locked' % self.path)
        elif not os.path.exists(self.unique_name):
            raise NotMyLock('%s is locked, but not by me' % self.path)
        os.unlink(self.unique_name)
        os.unlink(self.lock_file)

    def is_locked(self):
        if False:
            i = 10
            return i + 15
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        if False:
            while True:
                i = 10
        return self.is_locked() and os.path.exists(self.unique_name) and (os.stat(self.unique_name).st_nlink == 2)

    def break_lock(self):
        if False:
            print('Hello World!')
        if os.path.exists(self.lock_file):
            os.unlink(self.lock_file)