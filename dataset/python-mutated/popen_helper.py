"""Helper file for running the async data generation process in OSS."""
import contextlib
import multiprocessing
import multiprocessing.pool

def get_forkpool(num_workers, init_worker=None, closing=True):
    if False:
        for i in range(10):
            print('nop')
    pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker)
    return contextlib.closing(pool) if closing else pool

def get_threadpool(num_workers, init_worker=None, closing=True):
    if False:
        i = 10
        return i + 15
    pool = multiprocessing.pool.ThreadPool(processes=num_workers, initializer=init_worker)
    return contextlib.closing(pool) if closing else pool

class FauxPool(object):
    """Mimic a pool using for loops.

  This class is used in place of proper pools when true determinism is desired
  for testing or debugging.
  """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def map(self, func, iterable, chunksize=None):
        if False:
            return 10
        return [func(i) for i in iterable]

    def imap(self, func, iterable, chunksize=1):
        if False:
            for i in range(10):
                print('nop')
        for i in iterable:
            yield func(i)

    def close(self):
        if False:
            while True:
                i = 10
        pass

    def terminate(self):
        if False:
            i = 10
            return i + 15
        pass

    def join(self):
        if False:
            return 10
        pass

def get_fauxpool(num_workers, init_worker=None, closing=True):
    if False:
        for i in range(10):
            print('nop')
    pool = FauxPool(processes=num_workers, initializer=init_worker)
    return contextlib.closing(pool) if closing else pool

def worker_job():
    if False:
        for i in range(10):
            print('nop')
    return 'worker'