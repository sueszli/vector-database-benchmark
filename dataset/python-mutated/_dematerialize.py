from typing import Callable, Optional, TypeVar
from reactivex import Notification, Observable, abc
_T = TypeVar('_T')

def dematerialize_() -> Callable[[Observable[Notification[_T]]], Observable[_T]]:
    if False:
        return 10

    def dematerialize(source: Observable[Notification[_T]]) -> Observable[_T]:
        if False:
            return 10
        "Partially applied dematerialize operator.\n\n        Dematerializes the explicit notification values of an\n        observable sequence as implicit notifications.\n\n        Returns:\n            An observable sequence exhibiting the behavior\n            corresponding to the source sequence's notification values.\n        "

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                for i in range(10):
                    print('nop')

            def on_next(value: Notification[_T]) -> None:
                if False:
                    return 10
                return value.accept(observer)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return dematerialize
__all__ = ['dematerialize_']