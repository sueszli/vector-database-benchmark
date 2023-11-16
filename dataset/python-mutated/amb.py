from typing import TypeVar
from reactivex import Observable, never
from reactivex import operators as _
_T = TypeVar('_T')

def amb_(*sources: Observable[_T]) -> Observable[_T]:
    if False:
        while True:
            i = 10
    'Propagates the observable sequence that reacts first.\n\n    Example:\n        >>> winner = amb(xs, ys, zs)\n\n    Returns:\n        An observable sequence that surfaces any of the given sequences,\n        whichever reacted first.\n    '
    acc: Observable[_T] = never()

    def func(previous: Observable[_T], current: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10
        return _.amb(previous)(current)
    for source in sources:
        acc = func(acc, source)
    return acc
__all__ = ['amb_']