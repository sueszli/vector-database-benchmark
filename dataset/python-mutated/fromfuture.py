import asyncio
from asyncio import Future
from typing import Any, Optional, TypeVar, cast
from reactivex import Observable, abc
from reactivex.disposable import Disposable
_T = TypeVar('_T')

def from_future_(future: 'Future[_T]') -> Observable[_T]:
    if False:
        while True:
            i = 10
    'Converts a Future to an Observable sequence\n\n    Args:\n        future -- A Python 3 compatible future.\n            https://docs.python.org/3/library/asyncio-task.html#future\n\n    Returns:\n        An Observable sequence which wraps the existing future success\n        and failure.\n    '

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')

        def done(future: 'Future[_T]') -> None:
            if False:
                for i in range(10):
                    print('nop')
            try:
                value: Any = future.result()
            except Exception as ex:
                observer.on_error(ex)
            except asyncio.CancelledError as ex:
                observer.on_error(cast(Exception, ex))
            else:
                observer.on_next(value)
                observer.on_completed()
        future.add_done_callback(done)

        def dispose() -> None:
            if False:
                return 10
            if future:
                future.cancel()
        return Disposable(dispose)
    return Observable(subscribe)
__all__ = ['from_future_']