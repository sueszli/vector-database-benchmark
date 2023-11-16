from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.internal import ArgumentOutOfRangeException
_T = TypeVar('_T')

def skip_(count: int) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10
    if count < 0:
        raise ArgumentOutOfRangeException()

    def skip(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10
        'The skip operator.\n\n        Bypasses a specified number of elements in an observable sequence\n        and then returns the remaining elements.\n\n        Args:\n            source: The source observable.\n\n        Returns:\n            An observable sequence that contains the elements that occur\n            after the specified index in the input sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                return 10
            remaining = count

            def on_next(value: _T) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal remaining
                if remaining <= 0:
                    observer.on_next(value)
                else:
                    remaining -= 1
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return skip
__all__ = ['skip_']