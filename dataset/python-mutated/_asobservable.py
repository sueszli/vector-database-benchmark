from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def as_observable_() -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15

    def as_observable(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Hides the identity of an observable sequence.\n\n        Args:\n            source: Observable source to hide identity from.\n\n        Returns:\n            An observable sequence that hides the identity of the\n            source sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                while True:
                    i = 10
            return source.subscribe(observer, scheduler=scheduler)
        return Observable(subscribe)
    return as_observable
__all__ = ['as_observable_']