from asyncio import Future
from typing import Callable, Optional, TypeVar, Union
import reactivex
from reactivex import Observable, abc
from reactivex.disposable import SerialDisposable, SingleAssignmentDisposable
_T = TypeVar('_T')

def catch_handler(source: Observable[_T], handler: Callable[[Exception, Observable[_T]], Union[Observable[_T], 'Future[_T]']]) -> Observable[_T]:
    if False:
        return 10

    def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        d1 = SingleAssignmentDisposable()
        subscription = SerialDisposable()
        subscription.disposable = d1

        def on_error(exception: Exception) -> None:
            if False:
                while True:
                    i = 10
            try:
                result = handler(exception, source)
            except Exception as ex:
                observer.on_error(ex)
                return
            result = reactivex.from_future(result) if isinstance(result, Future) else result
            d = SingleAssignmentDisposable()
            subscription.disposable = d
            d.disposable = result.subscribe(observer, scheduler=scheduler)
        d1.disposable = source.subscribe(observer.on_next, on_error, observer.on_completed, scheduler=scheduler)
        return subscription
    return Observable(subscribe)

def catch_(handler: Union[Observable[_T], Callable[[Exception, Observable[_T]], Observable[_T]]]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def catch(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        "Continues an observable sequence that is terminated by an\n        exception with the next observable sequence.\n\n        Examples:\n            >>> op = catch(ys)\n            >>> op = catch(lambda ex, src: ys(ex))\n\n        Args:\n            handler: Second observable sequence used to produce\n                results when an error occurred in the first sequence, or an\n                exception handler function that returns an observable sequence\n                given the error and source observable that occurred in the\n                first sequence.\n\n        Returns:\n            An observable sequence containing the first sequence's\n            elements, followed by the elements of the handler sequence\n            in case an exception occurred.\n        "
        if callable(handler):
            return catch_handler(source, handler)
        else:
            return reactivex.catch(source, handler)
    return catch
__all__ = ['catch_']