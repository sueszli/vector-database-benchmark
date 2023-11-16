from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.typing import Predicate, PredicateIndexed
_T = TypeVar('_T')

def take_while_(predicate: Predicate[_T], inclusive: bool=False) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def take_while(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Returns elements from an observable sequence as long as a\n        specified condition is true.\n\n        Example:\n            >>> take_while(source)\n\n        Args:\n            source: The source observable to take from.\n\n        Returns:\n            An observable sequence that contains the elements from the\n            input sequence that occur before the element at which the\n            test no longer passes.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                print('Hello World!')
            running = True

            def on_next(value: _T):
                if False:
                    while True:
                        i = 10
                nonlocal running
                with source.lock:
                    if not running:
                        return
                    try:
                        running = predicate(value)
                    except Exception as exn:
                        observer.on_error(exn)
                        return
                if running:
                    observer.on_next(value)
                else:
                    if inclusive:
                        observer.on_next(value)
                    observer.on_completed()
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return take_while

def take_while_indexed_(predicate: PredicateIndexed[_T], inclusive: bool=False) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def take_while_indexed(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        "Returns elements from an observable sequence as long as a\n        specified condition is true. The element's index is used in the\n        logic of the predicate function.\n\n        Example:\n            >>> take_while(source)\n\n        Args:\n            source: Source observable to take from.\n\n        Returns:\n            An observable sequence that contains the elements from the\n            input sequence that occur before the element at which the\n            test no longer passes.\n        "

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            running = True
            i = 0

            def on_next(value: Any) -> None:
                if False:
                    i = 10
                    return i + 15
                nonlocal running, i
                with source.lock:
                    if not running:
                        return
                    try:
                        running = predicate(value, i)
                    except Exception as exn:
                        observer.on_error(exn)
                        return
                    else:
                        i += 1
                if running:
                    observer.on_next(value)
                else:
                    if inclusive:
                        observer.on_next(value)
                    observer.on_completed()
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return take_while_indexed
__all__ = ['take_while_', 'take_while_indexed_']