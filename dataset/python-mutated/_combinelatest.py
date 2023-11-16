from typing import Any, Callable
import reactivex
from reactivex import Observable

def combine_latest_(*others: Observable[Any]) -> Callable[[Observable[Any]], Observable[Any]]:
    if False:
        i = 10
        return i + 15

    def combine_latest(source: Observable[Any]) -> Observable[Any]:
        if False:
            i = 10
            return i + 15
        'Merges the specified observable sequences into one\n        observable sequence by creating a tuple whenever any\n        of the observable sequences produces an element.\n\n        Examples:\n            >>> obs = combine_latest(source)\n\n        Returns:\n            An observable sequence containing the result of combining\n            elements of the sources into a tuple.\n        '
        sources = (source,) + others
        return reactivex.combine_latest(*sources)
    return combine_latest
__all__ = ['combine_latest_']