from typing import Callable, Optional, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.typing import Predicate
_T = TypeVar('_T')

def count_(predicate: Optional[Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[int]]:
    if False:
        return 10
    if predicate:
        return compose(ops.filter(predicate), ops.count())

    def reducer(n: int, _: _T) -> int:
        if False:
            i = 10
            return i + 15
        return n + 1
    counter = ops.reduce(reducer, seed=0)
    return counter
__all__ = ['count_']