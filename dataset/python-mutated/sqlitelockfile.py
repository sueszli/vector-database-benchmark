from __future__ import absolute_import, division
import time
import os
try:
    unicode
except NameError:
    unicode = str
from . import LockBase, NotLocked, NotMyLock, LockTimeout, AlreadyLocked

class SQLiteLockFile(LockBase):
    """Demonstrate SQL-based locking."""
    testdb = None

    def __init__(self, path, threaded=True, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> lock = SQLiteLockFile('somefile')\n        >>> lock = SQLiteLockFile('somefile', threaded=False)\n        "
        LockBase.__init__(self, path, threaded, timeout)
        self.lock_file = unicode(self.lock_file)
        self.unique_name = unicode(self.unique_name)
        if SQLiteLockFile.testdb is None:
            import tempfile
            (_fd, testdb) = tempfile.mkstemp()
            os.close(_fd)
            os.unlink(testdb)
            del _fd, tempfile
            SQLiteLockFile.testdb = testdb
        import sqlite3
        self.connection = sqlite3.connect(SQLiteLockFile.testdb)
        c = self.connection.cursor()
        try:
            c.execute('create table locks(   lock_file varchar(32),   unique_name varchar(32))')
        except sqlite3.OperationalError:
            pass
        else:
            self.connection.commit()
            import atexit
            atexit.register(os.unlink, SQLiteLockFile.testdb)

    def acquire(self, timeout=None):
        if False:
            while True:
                i = 10
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout
        if timeout is None:
            wait = 0.1
        elif timeout <= 0:
            wait = 0
        else:
            wait = timeout / 10
        cursor = self.connection.cursor()
        while True:
            if not self.is_locked():
                cursor.execute('insert into locks  (lock_file, unique_name)  values  (?, ?)', (self.lock_file, self.unique_name))
                self.connection.commit()
                cursor.execute('select * from locks  where unique_name = ?', (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) > 1:
                    cursor.execute('delete from locks  where unique_name = ?', (self.unique_name,))
                    self.connection.commit()
                else:
                    return
            else:
                cursor.execute('select * from locks  where unique_name = ?', (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) == 1:
                    return
            if timeout is not None and time.time() > end_time:
                if timeout > 0:
                    raise LockTimeout('Timeout waiting to acquire lock for %s' % self.path)
                else:
                    raise AlreadyLocked('%s is already locked' % self.path)
            time.sleep(wait)

    def release(self):
        if False:
            print('Hello World!')
        if not self.is_locked():
            raise NotLocked('%s is not locked' % self.path)
        if not self.i_am_locking():
            raise NotMyLock('%s is locked, but not by me (by %s)' % (self.unique_name, self._who_is_locking()))
        cursor = self.connection.cursor()
        cursor.execute('delete from locks  where unique_name = ?', (self.unique_name,))
        self.connection.commit()

    def _who_is_locking(self):
        if False:
            return 10
        cursor = self.connection.cursor()
        cursor.execute('select unique_name from locks  where lock_file = ?', (self.lock_file,))
        return cursor.fetchone()[0]

    def is_locked(self):
        if False:
            return 10
        cursor = self.connection.cursor()
        cursor.execute('select * from locks  where lock_file = ?', (self.lock_file,))
        rows = cursor.fetchall()
        return not not rows

    def i_am_locking(self):
        if False:
            for i in range(10):
                print('nop')
        cursor = self.connection.cursor()
        cursor.execute('select * from locks  where lock_file = ?    and unique_name = ?', (self.lock_file, self.unique_name))
        return not not cursor.fetchall()

    def break_lock(self):
        if False:
            while True:
                i = 10
        cursor = self.connection.cursor()
        cursor.execute('delete from locks  where lock_file = ?', (self.lock_file,))
        self.connection.commit()