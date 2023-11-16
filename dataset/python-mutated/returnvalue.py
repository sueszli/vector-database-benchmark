from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.scheduler import CurrentThreadScheduler
_T = TypeVar('_T')

def return_value_(value: _T, scheduler: Optional[abc.SchedulerBase]=None) -> Observable[_T]:
    if False:
        while True:
            i = 10
    "Returns an observable sequence that contains a single element,\n    using the specified scheduler to send out observer messages.\n    There is an alias called 'just'.\n\n    Examples:\n        >>> res = return(42)\n        >>> res = return(42, rx.Scheduler.timeout)\n\n    Args:\n        value: Single element in the resulting observable sequence.\n\n    Returns:\n        An observable sequence containing the single specified\n        element.\n    "

    def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        _scheduler = scheduler or scheduler_ or CurrentThreadScheduler.singleton()

        def action(scheduler: abc.SchedulerBase, state: Any=None) -> None:
            if False:
                print('Hello World!')
            observer.on_next(value)
            observer.on_completed()
        return _scheduler.schedule(action)
    return Observable(subscribe)

def from_callable_(supplier: Callable[[], _T], scheduler: Optional[abc.SchedulerBase]=None) -> Observable[_T]:
    if False:
        i = 10
        return i + 15

    def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            return 10
        _scheduler = scheduler or scheduler_ or CurrentThreadScheduler.singleton()

        def action(_: abc.SchedulerBase, __: Any=None) -> None:
            if False:
                return 10
            nonlocal observer
            try:
                observer.on_next(supplier())
                observer.on_completed()
            except Exception as e:
                observer.on_error(e)
        return _scheduler.schedule(action)
    return Observable(subscribe)
__all__ = ['return_value_', 'from_callable_']