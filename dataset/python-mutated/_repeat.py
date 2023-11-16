import sys
from typing import Callable, Optional, TypeVar
import reactivex
from reactivex import Observable
from reactivex.internal.utils import infinite
_T = TypeVar('_T')

def repeat_(repeat_count: Optional[int]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10
    if repeat_count is None:
        repeat_count = sys.maxsize

    def repeat(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Repeats the observable sequence a specified number of times.\n        If the repeat count is not specified, the sequence repeats\n        indefinitely.\n\n        Examples:\n            >>> repeated = source.repeat()\n            >>> repeated = source.repeat(42)\n\n        Args:\n            source: The observable source to repeat.\n\n        Returns:\n            The observable sequence producing the elements of the given\n            sequence repeatedly.\n        '
        if repeat_count is None:
            gen = infinite()
        else:
            gen = range(repeat_count)
        return reactivex.defer(lambda _: reactivex.concat_with_iterable((source for _ in gen)))
    return repeat
__all = ['repeat']