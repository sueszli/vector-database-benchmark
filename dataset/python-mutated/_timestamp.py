from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generic, Optional, TypeVar
from reactivex import Observable, abc, defer, operators
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

@dataclass
class Timestamp(Generic[_T]):
    value: _T
    timestamp: datetime

def timestamp_(scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[Timestamp[_T]]]:
    if False:
        i = 10
        return i + 15

    def timestamp(source: Observable[_T]) -> Observable[Timestamp[_T]]:
        if False:
            print('Hello World!')
        'Records the timestamp for each value in an observable sequence.\n\n        Examples:\n            >>> timestamp(source)\n\n        Produces objects with attributes `value` and `timestamp`, where\n        value is the original value.\n\n        Args:\n            source: Observable source to timestamp.\n\n        Returns:\n            An observable sequence with timestamp information on values.\n        '

        def factory(scheduler_: Optional[abc.SchedulerBase]=None):
            if False:
                while True:
                    i = 10
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()

            def mapper(value: _T) -> Timestamp[_T]:
                if False:
                    return 10
                return Timestamp(value=value, timestamp=_scheduler.now)
            return source.pipe(operators.map(mapper))
        return defer(factory)
    return timestamp
__all__ = ['timestamp_']