from typing import Callable, Optional, Tuple, TypeVar
from reactivex import Observable, abc, compose
from reactivex import operators as ops
from reactivex import typing
_T = TypeVar('_T')

def skip_while_(predicate: typing.Predicate[_T]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def skip_while(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10
        "Bypasses elements in an observable sequence as long as a\n        specified condition is true and then returns the remaining\n        elements. The element's index is used in the logic of the\n        predicate function.\n\n        Example:\n            >>> skip_while(source)\n\n        Args:\n            source: The source observable to skip elements from.\n\n        Returns:\n            An observable sequence that contains the elements from the\n            input sequence starting at the first element in the linear\n            series that does not pass the test specified by predicate.\n        "

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                for i in range(10):
                    print('nop')
            running = False

            def on_next(value: _T):
                if False:
                    while True:
                        i = 10
                nonlocal running
                if not running:
                    try:
                        running = not predicate(value)
                    except Exception as exn:
                        observer.on_error(exn)
                        return
                if running:
                    observer.on_next(value)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return skip_while

def skip_while_indexed_(predicate: typing.PredicateIndexed[_T]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15

    def indexer(x: _T, i: int) -> Tuple[_T, int]:
        if False:
            return 10
        return (x, i)

    def skipper(x: Tuple[_T, int]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return predicate(*x)

    def mapper(x: Tuple[_T, int]) -> _T:
        if False:
            print('Hello World!')
        return x[0]
    return compose(ops.map_indexed(indexer), ops.skip_while(skipper), ops.map(mapper))
__all__ = ['skip_while_', 'skip_while_indexed_']