from typing import Any, Callable
from reactivex import Observable, compose
from reactivex import operators as ops

def is_empty_() -> Callable[[Observable[Any]], Observable[bool]]:
    if False:
        i = 10
        return i + 15
    'Determines whether an observable sequence is empty.\n\n    Returns:\n        An observable sequence containing a single element\n        determining whether the source sequence is empty.\n    '

    def mapper(b: bool) -> bool:
        if False:
            print('Hello World!')
        return not b
    return compose(ops.some(), ops.map(mapper))
__all__ = ['is_empty_']