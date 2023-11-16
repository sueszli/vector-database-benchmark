from typing import Any, Callable
import reactivex
from reactivex import Observable

def with_latest_from_(*sources: Observable[Any]) -> Callable[[Observable[Any]], Observable[Any]]:
    if False:
        return 10
    'With latest from operator.\n\n    Merges the specified observable sequences into one observable\n    sequence by creating a tuple only when the first\n    observable sequence produces an element. The observables can be\n    passed either as seperate arguments or as a list.\n\n    Examples:\n        >>> op = with_latest_from(obs1)\n        >>> op = with_latest_from(obs1, obs2, obs3)\n\n    Returns:\n        An observable sequence containing the result of combining\n    elements of the sources into a tuple.\n    '

    def with_latest_from(source: Observable[Any]) -> Observable[Any]:
        if False:
            i = 10
            return i + 15
        return reactivex.with_latest_from(source, *sources)
    return with_latest_from
__all__ = ['with_latest_from_']