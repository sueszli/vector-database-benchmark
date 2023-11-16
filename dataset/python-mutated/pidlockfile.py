""" Lockfile behaviour implemented via Unix PID files.
    """
from __future__ import absolute_import
import errno
import os
import time
from . import LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock, LockTimeout

class PIDLockFile(LockBase):
    """ Lockfile implemented as a Unix PID file.

    The lock file is a normal file named by the attribute `path`.
    A lock's PID file contains a single line of text, containing
    the process ID (PID) of the process that acquired the lock.

    >>> lock = PIDLockFile('somefile')
    >>> lock = PIDLockFile('somefile')
    """

    def __init__(self, path, threaded=False, timeout=None):
        if False:
            i = 10
            return i + 15
        LockBase.__init__(self, path, False, timeout)
        self.unique_name = self.path

    def read_pid(self):
        if False:
            i = 10
            return i + 15
        ' Get the PID from the lock file.\n            '
        return read_pid_from_pidfile(self.path)

    def is_locked(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test if the lock is currently held.\n\n            The lock is held if the PID file for this lock exists.\n\n            '
        return os.path.exists(self.path)

    def i_am_locking(self):
        if False:
            while True:
                i = 10
        ' Test if the lock is held by the current process.\n\n        Returns ``True`` if the current process ID matches the\n        number stored in the PID file.\n        '
        return self.is_locked() and os.getpid() == self.read_pid()

    def acquire(self, timeout=None):
        if False:
            print('Hello World!')
        ' Acquire the lock.\n\n        Creates the PID file for this lock, or raises an error if\n        the lock could not be acquired.\n        '
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout
        while True:
            try:
                write_pid_to_pidfile(self.path)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    if time.time() > end_time:
                        if timeout is not None and timeout > 0:
                            raise LockTimeout('Timeout waiting to acquire lock for %s' % self.path)
                        else:
                            raise AlreadyLocked('%s is already locked' % self.path)
                    time.sleep(timeout is not None and timeout / 10 or 0.1)
                else:
                    raise LockFailed('failed to create %s' % self.path)
            else:
                return

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        ' Release the lock.\n\n            Removes the PID file to release the lock, or raises an\n            error if the current process does not hold the lock.\n\n            '
        if not self.is_locked():
            raise NotLocked('%s is not locked' % self.path)
        if not self.i_am_locking():
            raise NotMyLock('%s is locked, but not by me' % self.path)
        remove_existing_pidfile(self.path)

    def break_lock(self):
        if False:
            return 10
        ' Break an existing lock.\n\n            Removes the PID file if it already exists, otherwise does\n            nothing.\n\n            '
        remove_existing_pidfile(self.path)

def read_pid_from_pidfile(pidfile_path):
    if False:
        while True:
            i = 10
    ' Read the PID recorded in the named PID file.\n\n        Read and return the numeric PID recorded as text in the named\n        PID file. If the PID file cannot be read, or if the content is\n        not a valid PID, return ``None``.\n\n        '
    pid = None
    try:
        pidfile = open(pidfile_path, 'r')
    except IOError:
        pass
    else:
        line = pidfile.readline().strip()
        try:
            pid = int(line)
        except ValueError:
            pass
        pidfile.close()
    return pid

def write_pid_to_pidfile(pidfile_path):
    if False:
        i = 10
        return i + 15
    ' Write the PID in the named PID file.\n\n        Get the numeric process ID (“PID”) of the current process\n        and write it to the named file as a line of text.\n\n        '
    open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    open_mode = 420
    pidfile_fd = os.open(pidfile_path, open_flags, open_mode)
    pidfile = os.fdopen(pidfile_fd, 'w')
    pid = os.getpid()
    pidfile.write('%s\n' % pid)
    pidfile.close()

def remove_existing_pidfile(pidfile_path):
    if False:
        for i in range(10):
            print('nop')
    " Remove the named PID file if it exists.\n\n        Removing a PID file that doesn't already exist puts us in the\n        desired state, so we ignore the condition if the file does not\n        exist.\n\n        "
    try:
        os.remove(pidfile_path)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            pass
        else:
            raise