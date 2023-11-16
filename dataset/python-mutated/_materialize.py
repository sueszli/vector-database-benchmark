from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.notification import Notification, OnCompleted, OnError, OnNext
_T = TypeVar('_T')

def materialize() -> Callable[[Observable[_T]], Observable[Notification[_T]]]:
    if False:
        while True:
            i = 10

    def materialize(source: Observable[_T]) -> Observable[Notification[_T]]:
        if False:
            return 10
        'Partially applied materialize operator.\n\n        Materializes the implicit notifications of an observable\n        sequence as explicit notification values.\n\n        Args:\n            source: Source observable to materialize.\n\n        Returns:\n            An observable sequence containing the materialized\n            notification values from the source sequence.\n        '

        def subscribe(observer: abc.ObserverBase[Notification[_T]], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                return 10

            def on_next(value: _T) -> None:
                if False:
                    i = 10
                    return i + 15
                observer.on_next(OnNext(value))

            def on_error(error: Exception) -> None:
                if False:
                    while True:
                        i = 10
                observer.on_next(OnError(error))
                observer.on_completed()

            def on_completed() -> None:
                if False:
                    while True:
                        i = 10
                observer.on_next(OnCompleted())
                observer.on_completed()
            return source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return materialize
__all__ = ['materialize']