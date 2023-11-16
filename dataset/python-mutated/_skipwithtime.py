from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def skip_with_time_(duration: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def skip_with_time(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        "Skips elements for the specified duration from the start of\n        the observable source sequence.\n\n        Args:\n            >>> res = skip_with_time(5.0)\n\n        Specifying a zero value for duration doesn't guarantee no\n        elements will be dropped from the start of the source sequence.\n        This is a side-effect of the asynchrony introduced by the\n        scheduler, where the action that causes callbacks from the\n        source sequence to be forwarded may not execute immediately,\n        despite the zero due time.\n\n        Errors produced by the source sequence are always forwarded to\n        the result sequence, even if the error occurs before the\n        duration.\n\n        Args:\n            duration: Duration for skipping elements from the start of\n            the sequence.\n\n        Returns:\n            An observable sequence with the elements skipped during the\n            specified duration from the start of the source sequence.\n        "

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None):
            if False:
                i = 10
                return i + 15
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            open = [False]

            def action(scheduler: abc.SchedulerBase, state: Any) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                open[0] = True
            t = _scheduler.schedule_relative(duration, action)

            def on_next(x: _T):
                if False:
                    i = 10
                    return i + 15
                if open[0]:
                    observer.on_next(x)
            d = source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler=scheduler_)
            return CompositeDisposable(t, d)
        return Observable(subscribe)
    return skip_with_time
__all__ = ['skip_with_time_']