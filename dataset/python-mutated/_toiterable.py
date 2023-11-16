from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def to_iterable_() -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        return 10

    def to_iterable(source: Observable[_T]) -> Observable[List[_T]]:
        if False:
            while True:
                i = 10
        'Creates an iterable from an observable sequence.\n\n        Returns:\n            An observable sequence containing a single element with an\n            iterable containing all the elements of the source\n            sequence.\n        '

        def subscribe(observer: abc.ObserverBase[List[_T]], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal source
            queue: List[_T] = []

            def on_next(item: _T):
                if False:
                    while True:
                        i = 10
                queue.append(item)

            def on_completed():
                if False:
                    return 10
                nonlocal queue
                observer.on_next(queue)
                queue = []
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return to_iterable
__all__ = ['to_iterable_']