from asyncio import Future
from typing import Callable, Optional, TypeVar, Union
from reactivex import Observable, abc, from_future, throw
from reactivex.scheduler import ImmediateScheduler
_T = TypeVar('_T')

def defer_(factory: Callable[[abc.SchedulerBase], Union[Observable[_T], 'Future[_T]']]) -> Observable[_T]:
    if False:
        while True:
            i = 10
    'Returns an observable sequence that invokes the specified factory\n    function whenever a new observer subscribes.\n\n    Example:\n        >>> res = defer(lambda scheduler: of(1, 2, 3))\n\n    Args:\n        observable_factory: Observable factory function to invoke for\n        each observer that subscribes to the resulting sequence. The\n        factory takes a single argument, the scheduler used.\n\n    Returns:\n        An observable sequence whose observers trigger an invocation\n        of the given observable factory function.\n    '

    def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        try:
            result = factory(scheduler or ImmediateScheduler.singleton())
        except Exception as ex:
            return throw(ex).subscribe(observer)
        result = from_future(result) if isinstance(result, Future) else result
        return result.subscribe(observer, scheduler=scheduler)
    return Observable(subscribe)
__all__ = ['defer_']