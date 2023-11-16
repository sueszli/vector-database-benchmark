__all__ = ['AuthProvider']
from contextlib import contextmanager
from sentry.auth import register, unregister

@contextmanager
def AuthProvider(name, cls):
    if False:
        for i in range(10):
            print('nop')
    register(name, cls)
    yield
    unregister(name, cls)