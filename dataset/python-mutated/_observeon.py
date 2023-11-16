from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.observer import ObserveOnObserver
_T = TypeVar('_T')

def observe_on_(scheduler: abc.SchedulerBase) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15

    def observe_on(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        'Wraps the source sequence in order to run its observer\n        callbacks on the specified scheduler.\n\n        This only invokes observer callbacks on a scheduler. In case\n        the subscription and/or unsubscription actions have\n        side-effects that require to be run on a scheduler, use\n        subscribe_on.\n\n        Args:\n            source: Source observable.\n\n\n        Returns:\n            Returns the source sequence whose observations happen on\n            the specified scheduler.\n        '

        def subscribe(observer: abc.ObserverBase[_T], subscribe_scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                while True:
                    i = 10
            return source.subscribe(ObserveOnObserver(scheduler, observer), scheduler=subscribe_scheduler)
        return Observable(subscribe)
    return observe_on
__all__ = ['observe_on_']