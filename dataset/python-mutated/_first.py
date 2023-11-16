from typing import Callable, Optional, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.typing import Predicate
from ._firstordefault import first_or_default_async_
_T = TypeVar('_T')

def first_(predicate: Optional[Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10
    'Returns the first element of an observable sequence that\n    satisfies the condition in the predicate if present else the first\n    item in the sequence.\n\n    Examples:\n        >>> res = res = first()(source)\n        >>> res = res = first(lambda x: x > 3)(source)\n\n    Args:\n        predicate -- [Optional] A predicate function to evaluate for\n            elements in the source sequence.\n\n    Returns:\n        A function that takes an observable source and returns an\n        observable sequence containing the first element in the\n        observable sequence that satisfies the condition in the predicate if\n        provided, else the first item in the sequence.\n    '
    if predicate:
        return compose(ops.filter(predicate), ops.first())
    return first_or_default_async_(False)
__all__ = ['first_']