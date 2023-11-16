from typing import Callable, Optional, Set, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def to_set_() -> Callable[[Observable[_T]], Observable[Set[_T]]]:
    if False:
        print('Hello World!')
    'Converts the observable sequence to a set.\n\n    Returns an observable sequence with a single value of a set\n    containing the values from the observable sequence.\n    '

    def to_set(source: Observable[_T]) -> Observable[Set[_T]]:
        if False:
            for i in range(10):
                print('nop')

        def subscribe(observer: abc.ObserverBase[Set[_T]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            s: Set[_T] = set()

            def on_completed() -> None:
                if False:
                    print('Hello World!')
                nonlocal s
                observer.on_next(s)
                s = set()
                observer.on_completed()
            return source.subscribe(s.add, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return to_set
__all__ = ['to_set_']