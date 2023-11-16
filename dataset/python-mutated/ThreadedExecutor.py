""" Threaded pool execution.

This can use Python3 native, or even Python2.7 backport, or has a Python2.6
stub that does not thread at all.
"""
from threading import RLock, current_thread
_use_threaded_executor = False

class NonThreadedPoolExecutor(object):

    def __init__(self, max_workers=None):
        if False:
            return 10
        self.results = []

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if False:
            return 10
        return False

    def submit(self, function, *args):
        if False:
            i = 10
            return i + 15
        self.results.append(function(*args))
        return self

def waitWorkers(workers):
    if False:
        while True:
            i = 10
    if workers:
        return iter(workers[0].results)
ThreadPoolExecutor = NonThreadedPoolExecutor
if _use_threaded_executor:
    try:
        from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, as_completed, wait

        def waitWorkers(workers):
            if False:
                i = 10
                return i + 15
            wait(workers, return_when=FIRST_EXCEPTION)
            for future in as_completed(workers):
                yield future.result()
    except ImportError:
        pass

def getThreadIdent():
    if False:
        return 10
    return current_thread()
assert RLock
assert ThreadPoolExecutor