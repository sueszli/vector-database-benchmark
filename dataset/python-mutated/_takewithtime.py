from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def take_with_time_(duration: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def take_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            return 10
        'Takes elements for the specified duration from the start of\n        the observable source sequence.\n\n        Example:\n            >>> res = take_with_time(source)\n\n        This operator accumulates a queue with a length enough to store\n        elements received during the initial duration window. As more\n        elements are received, elements older than the specified\n        duration are taken from the queue and produced on the result\n        sequence. This causes elements to be delayed with duration.\n\n        Args:\n            source: Source observable to take elements from.\n\n        Returns:\n            An observable sequence with the elements taken during the\n            specified duration from the start of the source sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()

            def action(scheduler: abc.SchedulerBase, state: Any=None):
                if False:
                    print('Hello World!')
                observer.on_completed()
            disp = _scheduler.schedule_relative(duration, action)
            return CompositeDisposable(disp, source.subscribe(observer, scheduler=scheduler_))
        return Observable(subscribe)
    return take_with_time
__all__ = ['take_with_time_']