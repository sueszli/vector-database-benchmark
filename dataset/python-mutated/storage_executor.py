import math
import multiprocessing
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from metaflow.exception import MetaflowException
if sys.version_info[:2] < (3, 7):
    from concurrent.futures.process import BrokenProcessPool
    BrokenStorageExecutorError = BrokenProcessPool
else:
    from concurrent.futures import BrokenExecutor as _BrokenExecutor
    BrokenStorageExecutorError = _BrokenExecutor

def _determine_effective_cpu_limit():
    if False:
        return 10
    'Calculate CPU limit (in number of cores) based on:\n\n    - /sys/fs/cgroup/cpu/cpu.max (if available, cgroup 2)\n    OR\n    - /sys/fs/cgroup/cpu/cpu.cfs_quota_us\n    - /sys/fs/cgroup/cpu/cpu.cfs_period_us\n\n    Returns:\n        > 0 if limit was successfully calculated\n        = 0 if we determined that there is no limit\n        -1 if we failed to determine the limit\n    '
    try:
        if platform.system() == 'Darwin':
            return 0
        elif platform.system() == 'Linux':
            with open('/sys/fs/cgroup/cpu.max', 'rb') as f:
                parts = f.read().decode('utf-8').split(' ')
                if len(parts) == 2:
                    if parts[0] == 'max':
                        return 0
                    return int(parts[0]) / int(parts[1])
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'rb') as f:
                quota = int(f.read())
                if quota == -1:
                    return 0
                if quota < 0:
                    return -1
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'rb') as f:
                period = int(f.read())
                if period <= 0:
                    return -1
                return quota / period
        else:
            return -1
    except Exception:
        return -1

def _noop_for_executor_warm_up():
    if False:
        i = 10
        return i + 15
    pass

def _compute_executor_max_workers():
    if False:
        print('Hello World!')
    min_processes = 4
    max_processes = 18
    effective_cpu_limit = _determine_effective_cpu_limit()

    def _bracket(min_v, v, max_v):
        if False:
            i = 10
            return i + 15
        assert min_v <= max_v
        if v < min_v:
            return min_v
        if v > max_v:
            return max_v
        return v
    if effective_cpu_limit < 0:
        processpool_max_workers = min_processes
    elif effective_cpu_limit == 0:
        processpool_max_workers = _bracket(min_processes, os.cpu_count() or 1, max_processes)
    else:
        processpool_max_workers = _bracket(min_processes, math.ceil(effective_cpu_limit), max_processes)
    threadpool_max_workers = processpool_max_workers + 4
    return (processpool_max_workers, threadpool_max_workers)

class StorageExecutor(object):
    """Thin wrapper around a ProcessPoolExecutor, or a ThreadPoolExecutor where
    the former may be unsafe.
    """

    def __init__(self, use_processes=False):
        if False:
            i = 10
            return i + 15
        (processpool_max_workers, threadpool_max_workers) = _compute_executor_max_workers()
        if use_processes:
            mp_start_method = multiprocessing.get_start_method(allow_none=True)
            if mp_start_method == 'spawn':
                self._executor = ProcessPoolExecutor(max_workers=processpool_max_workers)
            elif sys.version_info[:2] >= (3, 7):
                self._executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'), max_workers=processpool_max_workers)
            else:
                raise MetaflowException(msg="Cannot use ProcessPoolExecutor because Python version is older than 3.7 and multiprocess start method has been set to something other than 'spawn'")
        else:
            self._executor = ThreadPoolExecutor(max_workers=threadpool_max_workers)

    def warm_up(self):
        if False:
            print('Hello World!')
        self._executor.submit(_noop_for_executor_warm_up)

    def submit(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._executor.submit(*args, **kwargs)

def handle_executor_exceptions(func):
    if False:
        return 10
    '\n    Decorator for handling errors that come from an Executor. This decorator should\n    only be used on functions where executor errors are possible. I.e. the function\n    uses StorageExecutor.\n    '

    def inner_function(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return func(*args, **kwargs)
        except BrokenStorageExecutorError:
            sys.exit(1)
    return inner_function