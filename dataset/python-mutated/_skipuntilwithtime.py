from datetime import datetime
from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def skip_until_with_time_(start_time: typing.AbsoluteOrRelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def skip_until_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Skips elements from the observable source sequence until the\n        specified start time.\n\n        Errors produced by the source sequence are always forwarded to\n        the result sequence, even if the error occurs before the start\n        time.\n\n        Examples:\n            >>> res = source.skip_until_with_time(datetime)\n            >>> res = source.skip_until_with_time(5.0)\n\n        Args:\n            start_time: Time to start taking elements from the source\n                sequence. If this value is less than or equal to\n                `datetime.utcnow`, no elements will be skipped.\n\n        Returns:\n            An observable sequence with the elements skipped until the\n            specified start time.\n        '
        if isinstance(start_time, datetime):
            scheduler_method = 'schedule_absolute'
        else:
            scheduler_method = 'schedule_relative'

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None):
            if False:
                while True:
                    i = 10
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            open = [False]

            def on_next(x: _T) -> None:
                if False:
                    i = 10
                    return i + 15
                if open[0]:
                    observer.on_next(x)
            subscription = source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler_)

            def action(scheduler: abc.SchedulerBase, state: Any):
                if False:
                    return 10
                open[0] = True
            disp = getattr(_scheduler, scheduler_method)(start_time, action)
            return CompositeDisposable(disp, subscription)
        return Observable(subscribe)
    return skip_until_with_time
__all__ = ['skip_until_with_time_']