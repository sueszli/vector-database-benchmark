from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex import operators as ops
from reactivex.scheduler import TimeoutScheduler
from reactivex.subject import AsyncSubject
_T = TypeVar('_T')

def to_async_(func: Callable[..., _T], scheduler: Optional[abc.SchedulerBase]=None) -> Callable[..., Observable[_T]]:
    if False:
        i = 10
        return i + 15
    "Converts the function into an asynchronous function. Each\n    invocation of the resulting asynchronous function causes an\n    invocation of the original synchronous function on the specified\n    scheduler.\n\n    Examples:\n        res = reactivex.to_async(lambda x, y: x + y)(4, 3)\n        res = reactivex.to_async(lambda x, y: x + y, Scheduler.timeout)(4, 3)\n        res = reactivex.to_async(lambda x: log.debug(x), Scheduler.timeout)('hello')\n\n    Args:\n        func: Function to convert to an asynchronous function.\n        scheduler: [Optional] Scheduler to run the function on. If not\n            specified, defaults to Scheduler.timeout.\n\n    Returns:\n        Aynchronous function.\n    "
    _scheduler = scheduler or TimeoutScheduler.singleton()

    def wrapper(*args: Any) -> Observable[_T]:
        if False:
            print('Hello World!')
        subject: AsyncSubject[_T] = AsyncSubject()

        def action(scheduler: abc.SchedulerBase, state: Any=None) -> None:
            if False:
                while True:
                    i = 10
            try:
                result = func(*args)
            except Exception as ex:
                subject.on_error(ex)
                return
            subject.on_next(result)
            subject.on_completed()
        _scheduler.schedule(action)
        return subject.pipe(ops.as_observable())
    return wrapper
__all__ = ['to_async_']