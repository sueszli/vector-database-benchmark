from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def take_last_(count: int) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def take_last(source: Observable[_T]) -> Observable[_T]:
        if False:
            return 10
        'Returns a specified number of contiguous elements from the end of an\n        observable sequence.\n\n        Example:\n            >>> res = take_last(source)\n\n        This operator accumulates a buffer with a length enough to store\n        elements count elements. Upon completion of the source sequence, this\n        buffer is drained on the result sequence. This causes the elements to be\n        delayed.\n\n        Args:\n            source: Number of elements to take from the end of the source\n            sequence.\n\n        Returns:\n            An observable sequence containing the specified number of elements\n            from the end of the source sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            q: List[_T] = []

            def on_next(x: _T) -> None:
                if False:
                    print('Hello World!')
                q.append(x)
                if len(q) > count:
                    q.pop(0)

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                while q:
                    observer.on_next(q.pop(0))
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return take_last
__all__ = ['take_last_']