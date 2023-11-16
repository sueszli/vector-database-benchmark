from typing import Any, Callable, Type, TypeVar, Union, cast
from reactivex import Observable, compose
from reactivex import operators as ops
from reactivex.internal.utils import NotSet
from reactivex.typing import Accumulator
_T = TypeVar('_T')
_TState = TypeVar('_TState')

def reduce_(accumulator: Accumulator[_TState, _T], seed: Union[_TState, Type[NotSet]]=NotSet) -> Callable[[Observable[_T]], Observable[Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Applies an accumulator function over an observable sequence,\n    returning the result of the aggregation as a single element in the\n    result sequence. The specified seed value is used as the initial\n    accumulator value.\n\n    For aggregation behavior with incremental intermediate results, see\n    `scan()`.\n\n    Examples:\n        >>> res = reduce(lambda acc, x: acc + x)\n        >>> res = reduce(lambda acc, x: acc + x, 0)\n\n    Args:\n        accumulator: An accumulator function to be\n            invoked on each element.\n        seed: Optional initial accumulator value.\n\n    Returns:\n        An operator function that takes an observable source and returns\n        an observable sequence containing a single element with the\n        final accumulator value.\n    '
    if seed is not NotSet:
        seed_: _TState = cast(_TState, seed)
        scanner = ops.scan(accumulator, seed=seed_)
        return compose(scanner, ops.last_or_default(default_value=seed_))
    return compose(ops.scan(cast(Accumulator[_T, _T], accumulator)), ops.last())
__all__ = ['reduce_']