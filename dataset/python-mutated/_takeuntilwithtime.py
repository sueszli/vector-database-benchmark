from datetime import datetime
from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def take_until_with_time_(end_time: typing.AbsoluteOrRelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def take_until_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        'Takes elements for the specified duration until the specified end\n        time, using the specified scheduler to run timers.\n\n        Examples:\n            >>> res = take_until_with_time(source)\n\n        Args:\n            source: Source observale to take elements from.\n\n        Returns:\n            An observable sequence with the elements taken\n            until the specified end time.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()

            def action(scheduler: abc.SchedulerBase, state: Any=None):
                if False:
                    for i in range(10):
                        print('nop')
                observer.on_completed()
            if isinstance(end_time, datetime):
                task = _scheduler.schedule_absolute(end_time, action)
            else:
                task = _scheduler.schedule_relative(end_time, action)
            return CompositeDisposable(task, source.subscribe(observer, scheduler=scheduler_))
        return Observable(subscribe)
    return take_until_with_time
__all__ = ['take_until_with_time_']