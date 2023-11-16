import os
import multiprocessing
import threading
import platform
from enum import Enum, unique
from pathlib import Path
if platform.system().lower() != 'windows':
    import fcntl
else:
    fcntl = None

@unique
class LockContextType(Enum):
    """
    Enum to express the type of the lock
    """
    THREAD_LOCK = 1
    PROCESS_LOCK = 2
_LOCK_TYPE_MAPPING = {LockContextType.THREAD_LOCK: threading.Lock, LockContextType.PROCESS_LOCK: multiprocessing.Lock}

class LockContext(object):
    """
    Overview:
        Generate a LockContext in order to make sure the thread safety.

    Interfaces:
        ``__init__``, ``__enter__``, ``__exit__``

    Example:
        >>> with LockContext() as lock:
        >>>     print("Do something here.")
    """

    def __init__(self, type_: LockContextType=LockContextType.THREAD_LOCK):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Init the lock according to given type\n        '
        self.lock = _LOCK_TYPE_MAPPING[type_]()

    def acquire(self):
        if False:
            for i in range(10):
                print('nop')
        self.lock.acquire()

    def release(self):
        if False:
            return 10
        self.lock.release()

    def __enter__(self):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Entering the context and acquire lock\n        '
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Quiting the context and release lock\n        '
        self.lock.release()
rw_lock_mapping = {}

def get_rw_file_lock(name: str, op: str):
    if False:
        return 10
    "\n    Overview:\n        Get generated file lock with name and operator\n    Arguments:\n        - name (:obj:`str`) Lock's name.\n        - op (:obj:`str`) Assigned operator, i.e. ``read`` or ``write``.\n    Returns:\n        - (:obj:`RWLockFairD`) Generated rwlock\n    "
    assert op in ['read', 'write']
    try:
        from readerwriterlock import rwlock
    except ImportError:
        import sys
        from ditk import logging
        logging.warning('Please install readerwriterlock first, such as `pip3 install readerwriterlock`.')
        sys.exit(1)
    if name not in rw_lock_mapping:
        rw_lock_mapping[name] = rwlock.RWLockFairD()
    lock = rw_lock_mapping[name]
    if op == 'read':
        return lock.gen_rlock()
    elif op == 'write':
        return lock.gen_wlock()

class FcntlContext:

    def __init__(self, lock_path: str) -> None:
        if False:
            print('Hello World!')
        self.lock_path = lock_path
        self.f = None

    def __enter__(self) -> None:
        if False:
            i = 10
            return i + 15
        assert self.f is None, self.lock_path
        self.f = open(self.lock_path, 'w')
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self.f.close()
        self.f = None

def get_file_lock(name: str, op: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    if fcntl is None:
        return get_rw_file_lock(name, op)
    else:
        lock_name = name + '.lock'
        if not os.path.isfile(lock_name):
            try:
                Path(lock_name).touch()
            except Exception as e:
                pass
        return FcntlContext(lock_name)