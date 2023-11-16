from typing import Callable, TypeVar
import reactivex
from reactivex import Observable
_T = TypeVar('_T')

def start_with_(*args: _T) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def start_with(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Partially applied start_with operator.\n\n        Prepends a sequence of values to an observable sequence.\n\n        Example:\n            >>> start_with(source)\n\n        Returns:\n            The source sequence prepended with the specified values.\n        '
        start = reactivex.from_iterable(args)
        sequence = [start, source]
        return reactivex.concat(*sequence)
    return start_with
__all__ = ['start_with_']