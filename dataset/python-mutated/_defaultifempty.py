from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def default_if_empty_(default_value: Optional[_T]=None) -> Callable[[Observable[_T]], Observable[Optional[_T]]]:
    if False:
        i = 10
        return i + 15

    def default_if_empty(source: Observable[_T]) -> Observable[Optional[_T]]:
        if False:
            print('Hello World!')
        'Returns the elements of the specified sequence or the\n        specified value in a singleton sequence if the sequence is\n        empty.\n\n        Examples:\n            >>> obs = default_if_empty(source)\n\n        Args:\n            source: Source observable.\n\n        Returns:\n            An observable sequence that contains the specified default\n            value if the source is empty otherwise, the elements of the\n            source.\n        '

        def subscribe(observer: abc.ObserverBase[Optional[_T]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                i = 10
                return i + 15
            found = [False]

            def on_next(x: _T):
                if False:
                    return 10
                found[0] = True
                observer.on_next(x)

            def on_completed():
                if False:
                    return 10
                if not found[0]:
                    observer.on_next(default_value)
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return default_if_empty
__all__ = ['default_if_empty_']