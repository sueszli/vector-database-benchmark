from typing import Callable, Optional, TypeVar, cast
from reactivex import Observable, abc, compose
from reactivex import operators as ops
from reactivex.internal.exceptions import SequenceContainsNoElementsError
from reactivex.typing import Predicate
_T = TypeVar('_T')

def single_or_default_async_(has_default: bool=False, default_value: Optional[_T]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def single_or_default_async(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                i = 10
                return i + 15
            value = cast(_T, default_value)
            seen_value = False

            def on_next(x: _T):
                if False:
                    return 10
                nonlocal value, seen_value
                if seen_value:
                    observer.on_error(Exception('Sequence contains more than one element'))
                else:
                    value = x
                    seen_value = True

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                if not seen_value and (not has_default):
                    observer.on_error(SequenceContainsNoElementsError())
                else:
                    observer.on_next(value)
                    observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return single_or_default_async

def single_or_default_(predicate: Optional[Predicate[_T]]=None, default_value: _T=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')
    'Returns the only element of an observable sequence that matches\n    the predicate, or a default value if no such element exists this\n    method reports an exception if there is more than one element in the\n    observable sequence.\n\n    Examples:\n        >>> res = single_or_default()\n        >>> res = single_or_default(lambda x: x == 42)\n        >>> res = single_or_default(lambda x: x == 42, 0)\n        >>> res = single_or_default(None, 0)\n\n    Args:\n        predicate -- [Optional] A predicate function to evaluate for\n            elements in the source sequence.\n        default_value -- [Optional] The default value if the index is\n            outside the bounds of the source sequence.\n\n    Returns:\n        An observable Sequence containing the single element in the\n    observable sequence that satisfies the condition in the predicate,\n    or a default value if no such element exists.\n    '
    if predicate:
        return compose(ops.filter(predicate), ops.single_or_default(None, default_value))
    else:
        return single_or_default_async_(True, default_value)
__all__ = ['single_or_default_', 'single_or_default_async_']