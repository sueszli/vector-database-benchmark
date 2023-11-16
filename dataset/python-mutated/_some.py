from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex import operators as ops
from reactivex.typing import Predicate
_T = TypeVar('_T')

def some_(predicate: Optional[Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[bool]]:
    if False:
        for i in range(10):
            print('nop')

    def some(source: Observable[_T]) -> Observable[bool]:
        if False:
            return 10
        'Partially applied operator.\n\n        Determines whether some element of an observable sequence satisfies a\n        condition if present, else if some items are in the sequence.\n\n        Example:\n            >>> obs = some(source)\n\n        Args:\n            predicate -- A function to test each element for a condition.\n\n        Returns:\n            An observable sequence containing a single element\n            determining whether some elements in the source sequence\n            pass the test in the specified predicate if given, else if\n            some items are in the sequence.\n        '

        def subscribe(observer: abc.ObserverBase[bool], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                while True:
                    i = 10

            def on_next(_: _T):
                if False:
                    while True:
                        i = 10
                observer.on_next(True)
                observer.on_completed()

            def on_error():
                if False:
                    print('Hello World!')
                observer.on_next(False)
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_error, scheduler=scheduler)
        if predicate:
            return source.pipe(ops.filter(predicate), some_())
        return Observable(subscribe)
    return some
__all__ = ['some_']