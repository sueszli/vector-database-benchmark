"""
A threaded shared-memory scheduler

See local.py
"""
from __future__ import annotations
import atexit
import multiprocessing.pool
import sys
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import Lock, current_thread
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async
from dask.system import CPU_COUNT
from dask.typing import Key

def _thread_get_id():
    if False:
        return 10
    return current_thread().ident
main_thread = current_thread()
default_pool: Executor | None = None
pools: defaultdict[threading.Thread, dict[int, Executor]] = defaultdict(dict)
pools_lock = Lock()

def pack_exception(e, dumps):
    if False:
        return 10
    return (e, sys.exc_info()[2])

def get(dsk: Mapping, keys: Sequence[Key] | Key, cache=None, num_workers=None, pool=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Threaded cached implementation of dask.get\n\n    Parameters\n    ----------\n\n    dsk: dict\n        A dask dictionary specifying a workflow\n    keys: key or list of keys\n        Keys corresponding to desired data\n    num_workers: integer of thread count\n        The number of threads to use in the ThreadPool that will actually execute tasks\n    cache: dict-like (optional)\n        Temporary storage of results\n\n    Examples\n    --------\n    >>> inc = lambda x: x + 1\n    >>> add = lambda x, y: x + y\n    >>> dsk = {'x': 1, 'y': 2, 'z': (inc, 'x'), 'w': (add, 'z', 'y')}\n    >>> get(dsk, 'w')\n    4\n    >>> get(dsk, ['w', 'y'])\n    (4, 2)\n    "
    global default_pool
    pool = pool or config.get('pool', None)
    num_workers = num_workers or config.get('num_workers', None)
    thread = current_thread()
    with pools_lock:
        if pool is None:
            if num_workers is None and thread is main_thread:
                if default_pool is None:
                    default_pool = ThreadPoolExecutor(CPU_COUNT)
                    atexit.register(default_pool.shutdown)
                pool = default_pool
            elif thread in pools and num_workers in pools[thread]:
                pool = pools[thread][num_workers]
            else:
                pool = ThreadPoolExecutor(num_workers)
                atexit.register(pool.shutdown)
                pools[thread][num_workers] = pool
        elif isinstance(pool, multiprocessing.pool.Pool):
            pool = MultiprocessingPoolExecutor(pool)
    results = get_async(pool.submit, pool._max_workers, dsk, keys, cache=cache, get_id=_thread_get_id, pack_exception=pack_exception, **kwargs)
    with pools_lock:
        active_threads = set(threading.enumerate())
        if thread is not main_thread:
            for t in list(pools):
                if t not in active_threads:
                    for p in pools.pop(t).values():
                        p.shutdown()
    return results