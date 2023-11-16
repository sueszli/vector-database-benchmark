from typing import Any, Callable, Dict, List, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def skip_last_with_time_(duration: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10
    'Skips elements for the specified duration from the end of the\n    observable source sequence.\n\n    Example:\n        >>> res = skip_last_with_time(5.0)\n\n    This operator accumulates a queue with a length enough to store\n    elements received during the initial duration window. As more\n    elements are received, elements older than the specified duration\n    are taken from the queue and produced on the result sequence. This\n    causes elements to be delayed with duration.\n\n    Args:\n        duration: Duration for skipping elements from the end of the\n            sequence.\n        scheduler: Scheduler to use for time handling.\n\n    Returns:\n        An observable sequence with the elements skipped during the\n    specified duration from the end of the source sequence.\n    '

    def skip_last_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10
            nonlocal duration
            _scheduler: abc.SchedulerBase = scheduler or scheduler_ or TimeoutScheduler.singleton()
            duration = _scheduler.to_timedelta(duration)
            q: List[Dict[str, Any]] = []

            def on_next(x: _T) -> None:
                if False:
                    print('Hello World!')
                now = _scheduler.now
                q.append({'interval': now, 'value': x})
                while q and now - q[0]['interval'] >= duration:
                    observer.on_next(q.pop(0)['value'])

            def on_completed() -> None:
                if False:
                    while True:
                        i = 10
                now = _scheduler.now
                while q and now - q[0]['interval'] >= duration:
                    observer.on_next(q.pop(0)['value'])
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=_scheduler)
        return Observable(subscribe)
    return skip_last_with_time
__all__ = ['skip_last_with_time_']