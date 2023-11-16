from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, abc
_T = TypeVar('_T')

def skip_last_(count: int) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15

    def skip_last(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10
        'Bypasses a specified number of elements at the end of an\n        observable sequence.\n\n        This operator accumulates a queue with a length enough to store\n        the first `count` elements. As more elements are received,\n        elements are taken from the front of the queue and produced on\n        the result sequence. This causes elements to be delayed.\n\n        Args:\n            count: Number of elements to bypass at the end of the\n            source sequence.\n\n        Returns:\n            An observable sequence containing the source sequence\n            elements except for the bypassed ones at the end.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                for i in range(10):
                    print('nop')
            q: List[_T] = []

            def on_next(value: _T) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                front = None
                with source.lock:
                    q.append(value)
                    if len(q) > count:
                        front = q.pop(0)
                if front is not None:
                    observer.on_next(front)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return skip_last
__all__ = ['skip_last_']