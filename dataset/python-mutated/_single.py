from typing import Callable, Optional, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.typing import Predicate
_T = TypeVar('_T')

def single_(predicate: Optional[Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15
    'Returns the only element of an observable sequence that satisfies the\n    condition in the optional predicate, and reports an exception if there\n    is not exactly one element in the observable sequence.\n\n    Example:\n        >>> res = single()\n        >>> res = single(lambda x: x == 42)\n\n    Args:\n        predicate -- [Optional] A predicate function to evaluate for\n            elements in the source sequence.\n\n    Returns:\n        An observable sequence containing the single element in the\n        observable sequence that satisfies the condition in the predicate.\n    '
    if predicate:
        return compose(ops.filter(predicate), ops.single())
    else:
        return ops.single_or_default_async(False)
__all__ = ['single_']