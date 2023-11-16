from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def take_last_buffer_(count: int) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        for i in range(10):
            print('nop')

    def take_last_buffer(source: Observable[_T]) -> Observable[List[_T]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns an array with the specified number of contiguous\n        elements from the end of an observable sequence.\n\n        Example:\n            >>> res = take_last(source)\n\n        This operator accumulates a buffer with a length enough to\n        store elements count elements. Upon completion of the source\n        sequence, this buffer is drained on the result sequence. This\n        causes the elements to be delayed.\n\n        Args:\n            source: Source observable to take elements from.\n\n        Returns:\n            An observable sequence containing a single list with the\n            specified number of elements from the end of the source\n            sequence.\n        '

        def subscribe(observer: abc.ObserverBase[List[_T]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                while True:
                    i = 10
            q: List[_T] = []

            def on_next(x: _T) -> None:
                if False:
                    i = 10
                    return i + 15
                with source.lock:
                    q.append(x)
                    if len(q) > count:
                        q.pop(0)

            def on_completed() -> None:
                if False:
                    for i in range(10):
                        print('nop')
                observer.on_next(q)
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return take_last_buffer
__all__ = ['take_last_buffer_']