from threading import Lock
from fasteners.process_lock import InterProcessLock
from os.path import exists
from os import chmod

class ComboLock:
    """ A combined process and thread lock.

    Args:
        path (str): path to the lockfile for the lock
    """

    def __init__(self, path):
        if False:
            return 10
        if not exists(path):
            f = open(path, 'w+')
            f.close()
            chmod(path, 511)
        self.plock = InterProcessLock(path)
        self.tlock = Lock()

    def acquire(self, blocking=True):
        if False:
            for i in range(10):
                print('nop')
        " Acquire lock, locks thread and process lock.\n\n        Args:\n            blocking(bool): Set's blocking mode of acquire operation.\n                            Default True.\n\n        Returns: True if lock succeeded otherwise False\n        "
        if not blocking:
            tlocked = self.tlock.acquire(blocking=False)
            if not tlocked:
                return False
            plocked = self.plock.acquire(blocking=False)
            if not plocked:
                self.tlock.release()
                return False
        else:
            self.tlock.acquire()
            self.plock.acquire()
        return True

    def release(self):
        if False:
            return 10
        ' Release acquired lock. '
        self.plock.release()
        self.tlock.release()

    def __enter__(self):
        if False:
            print('Hello World!')
        ' Context handler, acquires lock in blocking mode. '
        self.acquire()
        return self

    def __exit__(self, _type, value, traceback):
        if False:
            while True:
                i = 10
        ' Releases the lock. '
        self.release()