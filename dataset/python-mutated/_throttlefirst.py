from datetime import datetime
from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def throttle_first_(window_duration: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def throttle_first(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10
        'Returns an observable that emits only the first item emitted\n        by the source Observable during sequential time windows of a\n        specified duration.\n\n        Args:\n            source: Source observable to throttle.\n\n        Returns:\n            An Observable that performs the throttle operation.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            duration = _scheduler.to_timedelta(window_duration or 0.0)
            if duration <= _scheduler.to_timedelta(0):
                raise ValueError('window_duration cannot be less or equal zero.')
            last_on_next: Optional[datetime] = None

            def on_next(x: _T) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal last_on_next
                emit = False
                now = _scheduler.now
                with source.lock:
                    if not last_on_next or now - last_on_next >= duration:
                        last_on_next = now
                        emit = True
                if emit:
                    observer.on_next(x)
            return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=_scheduler)
        return Observable(subscribe)
    return throttle_first
__all__ = ['throttle_first_']