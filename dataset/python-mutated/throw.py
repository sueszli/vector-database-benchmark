from typing import Any, Optional, Union
from reactivex import Observable, abc
from reactivex.scheduler import ImmediateScheduler

def throw_(exception: Union[str, Exception], scheduler: Optional[abc.SchedulerBase]=None) -> Observable[Any]:
    if False:
        i = 10
        return i + 15
    exception_ = exception if isinstance(exception, Exception) else Exception(exception)

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        _scheduler = scheduler or ImmediateScheduler.singleton()

        def action(scheduler: abc.SchedulerBase, state: Any) -> None:
            if False:
                print('Hello World!')
            observer.on_error(exception_)
        return _scheduler.schedule(action)
    return Observable(subscribe)
__all__ = ['throw_']