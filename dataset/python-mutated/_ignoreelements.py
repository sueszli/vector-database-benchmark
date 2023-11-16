from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.internal import noop
_T = TypeVar('_T')

def ignore_elements_() -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10
    'Ignores all elements in an observable sequence leaving only the\n    termination messages.\n\n    Returns:\n        An empty observable {Observable} sequence that signals\n        termination, successful or exceptional, of the source sequence.\n    '

    def ignore_elements(source: Observable[_T]) -> Observable[_T]:
        if False:
            return 10

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            return source.subscribe(noop, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return ignore_elements
__all__ = ['ignore_elements_']