import threading
from typing import Any, Callable
from celery.signals import task_failure, task_success
from django.core.signals import request_finished
from sentry import app
_cache = threading.local()

def request_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    if False:
        while True:
            i = 10
    '\n    A decorator to memoize functions on a per-request basis.\n    Arguments to the memoized function should NOT be objects\n    Use primitive types as arguments\n    '

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        if app.env.request is None:
            return func(*args, **kwargs)
        if not hasattr(_cache, 'items'):
            _cache.items = {}
        cache_key = (func, repr(args), repr(kwargs))
        if cache_key in _cache.items:
            rv = _cache.items[cache_key]
        else:
            rv = func(*args, **kwargs)
            _cache.items[cache_key] = rv
        return rv
    return wrapped

def clear_cache(**kwargs: Any) -> None:
    if False:
        return 10
    _cache.items = {}
request_finished.connect(clear_cache)
task_failure.connect(clear_cache)
task_success.connect(clear_cache)