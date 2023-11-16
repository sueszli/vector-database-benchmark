from typing import Callable, List, Optional, TypeVar, cast
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.internal.basic import identity
from reactivex.internal.exceptions import SequenceContainsNoElementsError
from reactivex.typing import Comparer
_T = TypeVar('_T')

def first_only(x: List[_T]) -> _T:
    if False:
        i = 10
        return i + 15
    if not x:
        raise SequenceContainsNoElementsError()
    return x[0]

def min_(comparer: Optional[Comparer[_T]]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10
    'The `min` operator.\n\n    Returns the minimum element in an observable sequence according to\n    the optional comparer else a default greater than less than check.\n\n    Examples:\n        >>> res = source.min()\n        >>> res = source.min(lambda x, y: x.value - y.value)\n\n    Args:\n        comparer: [Optional] Comparer used to compare elements.\n\n    Returns:\n        An observable sequence containing a single element\n        with the minimum element in the source sequence.\n    '
    return compose(ops.min_by(cast(Callable[[_T], _T], identity), comparer), ops.map(first_only))
__all__ = ['min_']