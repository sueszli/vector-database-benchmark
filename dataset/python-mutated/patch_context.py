from threading import Lock
from sentry.utils.imports import import_string

class PatchContext:

    def __init__(self, target, callback):
        if False:
            return 10
        (target, attr) = target.rsplit('.', 1)
        target = import_string(target)
        self.target = target
        self.attr = attr
        self.callback = callback
        self._lock = Lock()
        with self._lock:
            self.func = getattr(target, attr)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        self.unpatch()

    def patch(self):
        if False:
            print('Hello World!')
        with self._lock:
            func = getattr(self.target, self.attr)

            def wrapped(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                __traceback_hide__ = True
                return self.callback(self.func, *args, **kwargs)
            wrapped.__name__ = func.__name__
            if hasattr(func, '__doc__'):
                wrapped.__doc__ = func.__doc__
            if hasattr(func, '__module__'):
                wrapped.__module__ = func.__module__
            setattr(self.target, self.attr, wrapped)

    def unpatch(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            setattr(self.target, self.attr, self.func)