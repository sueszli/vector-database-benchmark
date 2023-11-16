from typing import Any, Callable, Dict, List, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def take_last_with_time_(duration: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def take_last_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Returns elements within the specified duration from the end\n        of the observable source sequence.\n\n        Example:\n            >>> res = take_last_with_time(source)\n\n        This operator accumulates a queue with a length enough to store\n        elements received during the initial duration window. As more\n        elements are received, elements older than the specified\n        duration are taken from the queue and produced on the result\n        sequence. This causes elements to be delayed with duration.\n\n        Args:\n            duration: Duration for taking elements from the end of the\n            sequence.\n\n        Returns:\n            An observable sequence with the elements taken during the\n            specified duration from the end of the source sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal duration
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            duration = _scheduler.to_timedelta(duration)
            q: List[Dict[str, Any]] = []

            def on_next(x: _T) -> None:
                if False:
                    return 10
                now = _scheduler.now
                q.append({'interval': now, 'value': x})
                while q and now - q[0]['interval'] >= duration:
                    q.pop(0)

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                now = _scheduler.now
                while q:
                    _next = q.pop(0)
                    if now - _next['interval'] <= duration:
                        observer.on_next(_next['value'])
                observer.on_completed()
            return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler_)
        return Observable(subscribe)
    return take_last_with_time
__all__ = ['take_last_with_time_']