from typing import Any, Callable, List, Optional, TypeVar
from reactivex import Observable, compose
from reactivex import operators as ops
_T = TypeVar('_T')

def buffer_(boundaries: Observable[Any]) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        return 10
    return compose(ops.window(boundaries), ops.flat_map(ops.to_list()))

def buffer_when_(closing_mapper: Callable[[], Observable[Any]]) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        return 10
    return compose(ops.window_when(closing_mapper), ops.flat_map(ops.to_list()))

def buffer_toggle_(openings: Observable[Any], closing_mapper: Callable[[Any], Observable[Any]]) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        while True:
            i = 10
    return compose(ops.window_toggle(openings, closing_mapper), ops.flat_map(ops.to_list()))

def buffer_with_count_(count: int, skip: Optional[int]=None) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        i = 10
        return i + 15
    'Projects each element of an observable sequence into zero or more\n    buffers which are produced based on element count information.\n\n    Examples:\n        >>> res = buffer_with_count(10)(xs)\n        >>> res = buffer_with_count(10, 1)(xs)\n\n    Args:\n        count: Length of each buffer.\n        skip: [Optional] Number of elements to skip between\n            creation of consecutive buffers. If not provided, defaults to\n            the count.\n\n    Returns:\n        A function that takes an observable source and returns an\n        observable sequence of buffers.\n    '

    def buffer_with_count(source: Observable[_T]) -> Observable[List[_T]]:
        if False:
            return 10
        nonlocal skip
        if skip is None:
            skip = count

        def mapper(value: Observable[_T]) -> Observable[List[_T]]:
            if False:
                while True:
                    i = 10
            return value.pipe(ops.to_list())

        def predicate(value: List[_T]) -> bool:
            if False:
                return 10
            return len(value) > 0
        return source.pipe(ops.window_with_count(count, skip), ops.flat_map(mapper), ops.filter(predicate))
    return buffer_with_count
__all__ = ['buffer_', 'buffer_with_count_', 'buffer_when_', 'buffer_toggle_']