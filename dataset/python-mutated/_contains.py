from typing import Callable, Optional, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex import typing
from reactivex.internal.basic import default_comparer
_T = TypeVar('_T')

def contains_(value: _T, comparer: Optional[typing.Comparer[_T]]=None) -> Callable[[Observable[_T]], Observable[bool]]:
    if False:
        print('Hello World!')
    comparer_ = comparer or default_comparer

    def predicate(v: _T) -> bool:
        if False:
            i = 10
            return i + 15
        return comparer_(v, value)
    return compose(ops.filter(predicate), ops.some())
__all__ = ['contains_']