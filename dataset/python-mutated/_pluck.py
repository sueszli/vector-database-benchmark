from typing import Any, Callable, Dict, TypeVar
from reactivex import Observable
from reactivex import operators as ops
_TKey = TypeVar('_TKey')
_TValue = TypeVar('_TValue')

def pluck_(key: _TKey) -> Callable[[Observable[Dict[_TKey, _TValue]]], Observable[_TValue]]:
    if False:
        while True:
            i = 10
    'Retrieves the value of a specified key using dict-like access (as in\n    element[key]) from all elements in the Observable sequence.\n\n    Args:\n        key: The key to pluck.\n\n    Returns a new Observable {Observable} sequence of key values.\n\n    To pluck an attribute of each element, use pluck_attr.\n    '

    def mapper(x: Dict[_TKey, _TValue]) -> _TValue:
        if False:
            while True:
                i = 10
        return x[key]
    return ops.map(mapper)

def pluck_attr_(prop: str) -> Callable[[Observable[Any]], Observable[Any]]:
    if False:
        while True:
            i = 10
    'Retrieves the value of a specified property (using getattr) from\n    all elements in the Observable sequence.\n\n    Args:\n        property: The property to pluck.\n\n    Returns a new Observable {Observable} sequence of property values.\n\n    To pluck values using dict-like access (as in element[key]) on each\n    element, use pluck.\n    '
    return ops.map(lambda x: getattr(x, prop))
__all__ = ['pluck_', 'pluck_attr_']