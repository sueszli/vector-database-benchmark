import functools
from typing import Any, Callable, Optional
from django.utils.functional import LazyObject, SimpleLazyObject, empty

def lazy_no_retry(func: Callable) -> SimpleLazyObject:
    if False:
        return 10
    'Wrap SimpleLazyObject while ensuring it is never re-evaluated on failure.\n\n    Wraps a given function into a ``SimpleLazyObject`` class while ensuring\n    if ``func`` fails, then ``func`` is never invoked again.\n\n    This mitigates an issue where an expensive ``func`` can be rerun for\n    each GraphQL resolver instead of flagging it as rejected/failed.\n    '
    error: Optional[Exception] = None

    @functools.wraps(func)
    def _wrapper():
        if False:
            return 10
        nonlocal error
        if error:
            raise error
        try:
            return func()
        except Exception as exc:
            error = exc
            raise
    return SimpleLazyObject(_wrapper)

def unwrap_lazy(obj: LazyObject) -> Any:
    if False:
        print('Hello World!')
    'Return the value of a given ``LazyObject``.'
    if obj._wrapped is empty:
        obj._setup()
    return obj._wrapped