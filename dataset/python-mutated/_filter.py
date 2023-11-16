from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.typing import Predicate, PredicateIndexed
_T = TypeVar('_T')

def filter_(predicate: Predicate[_T]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10

    def filter(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Partially applied filter operator.\n\n        Filters the elements of an observable sequence based on a\n        predicate.\n\n        Example:\n            >>> filter(source)\n\n        Args:\n            source: Source observable to filter.\n\n        Returns:\n            A filtered observable sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]) -> abc.DisposableBase:
            if False:
                return 10

            def on_next(value: _T):
                if False:
                    i = 10
                    return i + 15
                try:
                    should_run = predicate(value)
                except Exception as ex:
                    observer.on_error(ex)
                    return
                if should_run:
                    observer.on_next(value)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return filter

def filter_indexed_(predicate_indexed: Optional[PredicateIndexed[_T]]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def filter_indexed(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        "Partially applied indexed filter operator.\n\n        Filters the elements of an observable sequence based on a\n        predicate by incorporating the element's index.\n\n        Example:\n            >>> filter_indexed(source)\n\n        Args:\n            source: Source observable to filter.\n\n        Returns:\n            A filtered observable sequence.\n        "

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]):
            if False:
                return 10
            count = 0

            def on_next(value: _T):
                if False:
                    i = 10
                    return i + 15
                nonlocal count
                should_run = True
                if predicate_indexed:
                    try:
                        should_run = predicate_indexed(value, count)
                    except Exception as ex:
                        observer.on_error(ex)
                        return
                    else:
                        count += 1
                if should_run:
                    observer.on_next(value)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return filter_indexed
__all__ = ['filter_', 'filter_indexed_']