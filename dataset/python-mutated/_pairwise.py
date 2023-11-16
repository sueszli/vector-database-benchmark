from typing import Callable, Optional, Tuple, TypeVar, cast
from reactivex import Observable, abc
_T = TypeVar('_T')

def pairwise_() -> Callable[[Observable[_T]], Observable[Tuple[_T, _T]]]:
    if False:
        for i in range(10):
            print('nop')

    def pairwise(source: Observable[_T]) -> Observable[Tuple[_T, _T]]:
        if False:
            for i in range(10):
                print('nop')
        'Partially applied pairwise operator.\n\n        Returns a new observable that triggers on the second and\n        subsequent triggerings of the input observable. The Nth\n        triggering of the input observable passes the arguments from\n        the N-1th and Nth triggering as a pair. The argument passed to\n        the N-1th triggering is held in hidden internal state until the\n        Nth triggering occurs.\n\n        Returns:\n            An observable that triggers on successive pairs of\n            observations from the input observable as an array.\n        '

        def subscribe(observer: abc.ObserverBase[Tuple[_T, _T]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                print('Hello World!')
            has_previous = False
            previous: _T = cast(_T, None)

            def on_next(x: _T) -> None:
                if False:
                    print('Hello World!')
                nonlocal has_previous, previous
                pair = None
                with source.lock:
                    if has_previous:
                        pair = (previous, x)
                    else:
                        has_previous = True
                    previous = x
                if pair:
                    observer.on_next(pair)
            return source.subscribe(on_next, observer.on_error, observer.on_completed)
        return Observable(subscribe)
    return pairwise
__all__ = ['pairwise_']