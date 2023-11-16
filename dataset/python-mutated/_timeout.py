from asyncio import Future
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar, Union
from reactivex import Observable, abc, from_future, throw, typing
from reactivex.disposable import CompositeDisposable, SerialDisposable, SingleAssignmentDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def timeout_(duetime: typing.AbsoluteOrRelativeTime, other: Optional[Union[Observable[_T], 'Future[_T]']]=None, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')
    other = other or throw(Exception('Timeout'))
    if isinstance(other, Future):
        obs = from_future(other)
    else:
        obs = other

    def timeout(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        'Returns the source observable sequence or the other observable\n        sequence if duetime elapses.\n\n        Examples:\n            >>> res = timeout(source)\n\n        Args:\n            source: Source observable to timeout\n\n        Returns:\n            An observable sequence switching to the other sequence in\n            case of a timeout.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                i = 10
                return i + 15
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            switched = [False]
            _id = [0]
            original = SingleAssignmentDisposable()
            subscription = SerialDisposable()
            timer = SerialDisposable()
            subscription.disposable = original

            def create_timer() -> None:
                if False:
                    return 10
                my_id = _id[0]

                def action(scheduler: abc.SchedulerBase, state: Any=None):
                    if False:
                        return 10
                    switched[0] = _id[0] == my_id
                    timer_wins = switched[0]
                    if timer_wins:
                        subscription.disposable = obs.subscribe(observer, scheduler=scheduler)
                if isinstance(duetime, datetime):
                    timer.disposable = _scheduler.schedule_absolute(duetime, action)
                else:
                    timer.disposable = _scheduler.schedule_relative(duetime, action)
            create_timer()

            def on_next(value: _T) -> None:
                if False:
                    print('Hello World!')
                send_wins = not switched[0]
                if send_wins:
                    _id[0] += 1
                    observer.on_next(value)
                    create_timer()

            def on_error(error: Exception) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                on_error_wins = not switched[0]
                if on_error_wins:
                    _id[0] += 1
                    observer.on_error(error)

            def on_completed() -> None:
                if False:
                    return 10
                on_completed_wins = not switched[0]
                if on_completed_wins:
                    _id[0] += 1
                    observer.on_completed()
            original.disposable = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler_)
            return CompositeDisposable(subscription, timer)
        return Observable(subscribe)
    return timeout