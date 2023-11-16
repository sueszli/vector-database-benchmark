from typing import Callable, Optional, TypeVar, cast
from reactivex import Observable, abc
from reactivex.internal.exceptions import ArgumentOutOfRangeException
_T = TypeVar('_T')

def element_at_or_default_(index: int, has_default: bool=False, default_value: Optional[_T]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')
    if index < 0:
        raise ArgumentOutOfRangeException()

    def element_at_or_default(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10
            index_ = index

            def on_next(x: _T) -> None:
                if False:
                    while True:
                        i = 10
                nonlocal index_
                found = False
                with source.lock:
                    if index_:
                        index_ -= 1
                    else:
                        found = True
                if found:
                    observer.on_next(x)
                    observer.on_completed()

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                if not has_default:
                    observer.on_error(ArgumentOutOfRangeException())
                else:
                    observer.on_next(cast(_T, default_value))
                    observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return element_at_or_default
__all__ = ['element_at_or_default_']