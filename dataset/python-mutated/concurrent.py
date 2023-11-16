"""
Thin wrappers around `concurrent.futures`.
"""
from contextlib import contextmanager
from operator import length_hint
from os import cpu_count
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
__author__ = {'github.com/': ['casperdcl']}
__all__ = ['thread_map', 'process_map']

@contextmanager
def ensure_lock(tqdm_class, lock_name=''):
    if False:
        print('Hello World!')
    "get (create if necessary) and then restore `tqdm_class`'s lock"
    old_lock = getattr(tqdm_class, '_lock', None)
    lock = old_lock or tqdm_class.get_lock()
    lock = getattr(lock, lock_name, lock)
    tqdm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tqdm_class._lock
    else:
        tqdm_class.set_lock(old_lock)

def _executor_map(PoolExecutor, fn, *iterables, **tqdm_kwargs):
    if False:
        return 10
    '\n    Implementation of `thread_map` and `process_map`.\n\n    Parameters\n    ----------\n    tqdm_class  : [default: tqdm.auto.tqdm].\n    max_workers  : [default: min(32, cpu_count() + 4)].\n    chunksize  : [default: 1].\n    lock_name  : [default: "":str].\n    '
    kwargs = tqdm_kwargs.copy()
    if 'total' not in kwargs:
        kwargs['total'] = length_hint(iterables[0])
    tqdm_class = kwargs.pop('tqdm_class', tqdm_auto)
    max_workers = kwargs.pop('max_workers', min(32, cpu_count() + 4))
    chunksize = kwargs.pop('chunksize', 1)
    lock_name = kwargs.pop('lock_name', '')
    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        with PoolExecutor(max_workers=max_workers, initializer=tqdm_class.set_lock, initargs=(lk,)) as ex:
            return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))

def thread_map(fn, *iterables, **tqdm_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent of `list(map(fn, *iterables))`\n    driven by `concurrent.futures.ThreadPoolExecutor`.\n\n    Parameters\n    ----------\n    tqdm_class  : optional\n        `tqdm` class to use for bars [default: tqdm.auto.tqdm].\n    max_workers  : int, optional\n        Maximum number of workers to spawn; passed to\n        `concurrent.futures.ThreadPoolExecutor.__init__`.\n        [default: max(32, cpu_count() + 4)].\n    '
    from concurrent.futures import ThreadPoolExecutor
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)

def process_map(fn, *iterables, **tqdm_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent of `list(map(fn, *iterables))`\n    driven by `concurrent.futures.ProcessPoolExecutor`.\n\n    Parameters\n    ----------\n    tqdm_class  : optional\n        `tqdm` class to use for bars [default: tqdm.auto.tqdm].\n    max_workers  : int, optional\n        Maximum number of workers to spawn; passed to\n        `concurrent.futures.ProcessPoolExecutor.__init__`.\n        [default: min(32, cpu_count() + 4)].\n    chunksize  : int, optional\n        Size of chunks sent to worker processes; passed to\n        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].\n    lock_name  : str, optional\n        Member of `tqdm_class.get_lock()` to use [default: mp_lock].\n    '
    from concurrent.futures import ProcessPoolExecutor
    if iterables and 'chunksize' not in tqdm_kwargs:
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len > 1000:
            from warnings import warn
            warn('Iterable length %d > 1000 but `chunksize` is not set. This may seriously degrade multiprocess performance. Set `chunksize=1` or more.' % longest_iterable_len, TqdmWarning, stacklevel=2)
    if 'lock_name' not in tqdm_kwargs:
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs['lock_name'] = 'mp_lock'
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)