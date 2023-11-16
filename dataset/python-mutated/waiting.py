import sys
from contextlib import contextmanager
from threading import Semaphore
import six
from mock import patch

class WaitResult(object):
    sentinel = object()
    res = sentinel
    exc_info = None

    class NotReady(Exception):
        pass

    @property
    def has_result(self):
        if False:
            for i in range(10):
                print('nop')
        return self.res is not self.sentinel or self.exc_info is not None

    def send(self, res, exc_info):
        if False:
            return 10
        if not self.has_result:
            self.res = res
            self.exc_info = exc_info

    def get(self):
        if False:
            print('Hello World!')
        if not self.has_result:
            raise WaitResult.NotReady()
        if self.exc_info is not None:
            six.reraise(*self.exc_info)
        return self.res

@contextmanager
def wait_for_call(obj, target, callback=None):
    if False:
        i = 10
        return i + 15
    sem = Semaphore(0)
    result = WaitResult()
    unpatched = getattr(obj, target)

    def maybe_release(args, kwargs, res, exc_info):
        if False:
            for i in range(10):
                print('nop')
        should_release = True
        if callable(callback):
            should_release = callback(args, kwargs, res, exc_info)
        if should_release:
            result.send(res, exc_info)
            sem.release()

    def wraps(*args, **kwargs):
        if False:
            while True:
                i = 10
        res = None
        exc_info = None
        try:
            res = unpatched(*args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
        maybe_release(args, kwargs, res, exc_info)
        if exc_info is not None:
            six.reraise(*exc_info)
        return res
    with patch.object(obj, target, new=wraps):
        yield result
        sem.acquire()