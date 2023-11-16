from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex import operators as ops
from reactivex import typing
from reactivex.internal.exceptions import SequenceContainsNoElementsError
_T = TypeVar('_T')

def last_or_default_async(source: Observable[_T], has_default: bool=False, default_value: Optional[_T]=None) -> Observable[Optional[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def subscribe(observer: abc.ObserverBase[Optional[_T]], scheduler: Optional[abc.SchedulerBase]=None):
        if False:
            while True:
                i = 10
        value = [default_value]
        seen_value = [False]

        def on_next(x: _T) -> None:
            if False:
                print('Hello World!')
            value[0] = x
            seen_value[0] = True

        def on_completed():
            if False:
                return 10
            if not seen_value[0] and (not has_default):
                observer.on_error(SequenceContainsNoElementsError())
            else:
                observer.on_next(value[0])
                observer.on_completed()
        return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
    return Observable(subscribe)

def last_or_default(default_value: Optional[_T]=None, predicate: Optional[typing.Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[Any]]:
    if False:
        i = 10
        return i + 15

    def last_or_default(source: Observable[Any]) -> Observable[Any]:
        if False:
            while True:
                i = 10
        'Return last or default element.\n\n        Examples:\n            >>> res = _last_or_default(source)\n\n        Args:\n            source: Observable sequence to get the last item from.\n\n        Returns:\n            Observable sequence containing the last element in the\n            observable sequence.\n        '
        if predicate:
            return source.pipe(ops.filter(predicate), ops.last_or_default(default_value))
        return last_or_default_async(source, True, default_value)
    return last_or_default
__all__ = ['last_or_default', 'last_or_default_async']