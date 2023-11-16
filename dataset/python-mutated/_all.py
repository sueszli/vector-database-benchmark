from typing import Callable, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.typing import Predicate
_T = TypeVar('_T')

def all_(predicate: Predicate[_T]) -> Callable[[Observable[_T]], Observable[bool]]:
    if False:
        for i in range(10):
            print('nop')

    def filter(v: _T):
        if False:
            for i in range(10):
                print('nop')
        return not predicate(v)

    def mapping(b: bool) -> bool:
        if False:
            while True:
                i = 10
        return not b
    return compose(ops.filter(filter), ops.some(), ops.map(mapping))
__all__ = ['all_']