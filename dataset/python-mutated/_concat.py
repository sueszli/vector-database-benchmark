from typing import Callable, TypeVar
import reactivex
from reactivex import Observable
_T = TypeVar('_T')

def concat_(*sources: Observable[_T]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10

    def concat(source: Observable[_T]) -> Observable[_T]:
        if False:
            return 10
        'Concatenates all the observable sequences.\n\n        Examples:\n            >>> op = concat(xs, ys, zs)\n\n        Returns:\n            An operator function that takes one or more observable sources and\n            returns an observable sequence that contains the elements of\n            each given sequence, in sequential order.\n        '
        return reactivex.concat(source, *sources)
    return concat
__all__ = ['concat_']