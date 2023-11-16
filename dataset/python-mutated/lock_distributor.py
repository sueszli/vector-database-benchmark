"""LockDistributor for creating and managing a set of locks"""
import multiprocessing
import multiprocessing.managers
import threading
from enum import Enum, auto
from typing import Dict, Optional, Set, cast

class LockChain:
    """Wrapper class for acquiring multiple locks in the same order to prevent dead locks
    Can be used with `with` statement"""

    def __init__(self, lock_mapping: Dict[str, threading.Lock]):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        lock_mapping : Dict[str, threading.Lock]\n            Dictionary of locks with keys being used as generating reproduciable order for aquiring and releasing locks.\n        '
        self._locks = [value for (_, value) in sorted(lock_mapping.items())]

    def acquire(self) -> None:
        if False:
            return 10
        'Aquire all locks in the LockChain'
        for lock in self._locks:
            lock.acquire()

    def release(self) -> None:
        if False:
            return 10
        'Release all locks in the LockChain'
        for lock in self._locks:
            lock.release()

    def __enter__(self) -> 'LockChain':
        if False:
            while True:
                i = 10
        self.acquire()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        if False:
            while True:
                i = 10
        self.release()

class LockDistributorType(Enum):
    """Types of LockDistributor"""
    THREAD = auto()
    PROCESS = auto()

class LockDistributor:
    """Dynamic lock distributor that supports threads and processes.
    In the case of processes, both manager(server process) or shared memory can be used.
    """
    _lock_type: LockDistributorType
    _manager: Optional[multiprocessing.managers.SyncManager]
    _dict_lock: threading.Lock
    _locks: Dict[str, threading.Lock]

    def __init__(self, lock_type: LockDistributorType=LockDistributorType.THREAD, manager: Optional[multiprocessing.managers.SyncManager]=None):
        if False:
            for i in range(10):
                print('nop')
        '[summary]\n\n        Parameters\n        ----------\n        lock_type : LockDistributorType, optional\n            Whether locking with threads or processes, by default LockDistributorType.THREAD\n        manager : Optional[multiprocessing.managers.SyncManager], optional\n            Optional process sync mananger for creating proxy locks, by default None\n        '
        self._lock_type = lock_type
        self._manager = manager
        self._dict_lock = self._create_new_lock()
        self._locks = self._manager.dict() if self._lock_type == LockDistributorType.PROCESS and self._manager is not None else dict()

    def _create_new_lock(self) -> threading.Lock:
        if False:
            print('Hello World!')
        'Create a new lock based on lock type\n\n        Returns\n        -------\n        threading.Lock\n            Newly created lock\n        '
        if self._lock_type == LockDistributorType.THREAD:
            return threading.Lock()
        return self._manager.Lock() if self._manager is not None else cast(threading.Lock, multiprocessing.Lock())

    def get_lock(self, key: str) -> threading.Lock:
        if False:
            return 10
        'Retrieve a lock associating with the key\n        If the lock does not exist, a new lock will be created.\n\n        Parameters\n        ----------\n        key : Key for retrieving the lock\n\n        Returns\n        -------\n        threading.Lock\n            Lock associated with the key\n        '
        with self._dict_lock:
            if key not in self._locks:
                self._locks[key] = self._create_new_lock()
            return self._locks[key]

    def get_locks(self, keys: Set[str]) -> Dict[str, threading.Lock]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve a list of locks associating with keys\n\n        Parameters\n        ----------\n        keys : Set[str]\n            Set of keys for retrieving the locks\n\n        Returns\n        -------\n        Dict[str, threading.Lock]\n            Dictionary mapping keys to locks\n        '
        lock_mapping = dict()
        for key in keys:
            lock_mapping[key] = self.get_lock(key)
        return lock_mapping

    def get_lock_chain(self, keys: Set[str]) -> LockChain:
        if False:
            print('Hello World!')
        'Similar to get_locks, but retrieves a LockChain object instead of a dictionary\n\n        Parameters\n        ----------\n        keys : Set[str]\n            Set of keys for retrieving the locks\n\n        Returns\n        -------\n        LockChain\n            LockChain object containing all the locks associated with keys\n        '
        return LockChain(self.get_locks(keys))