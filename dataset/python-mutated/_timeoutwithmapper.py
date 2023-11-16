from typing import Any, Callable, Optional, TypeVar
import reactivex
from reactivex import Observable, abc
from reactivex.disposable import CompositeDisposable, SerialDisposable, SingleAssignmentDisposable
_T = TypeVar('_T')

def timeout_with_mapper_(first_timeout: Optional[Observable[_T]]=None, timeout_duration_mapper: Optional[Callable[[Any], Observable[Any]]]=None, other: Optional[Observable[_T]]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15
    'Returns the source observable sequence, switching to the other\n    observable sequence if a timeout is signaled.\n\n        res = timeout_with_mapper(reactivex.timer(500))\n        res = timeout_with_mapper(reactivex.timer(500), lambda x: reactivex.timer(200))\n        res = timeout_with_mapper(\n            reactivex.timer(500),\n            lambda x: reactivex.timer(200)),\n            reactivex.return_value(42)\n        )\n\n    Args:\n        first_timeout -- [Optional] Observable sequence that represents the\n            timeout for the first element. If not provided, this defaults to\n            reactivex.never().\n        timeout_duration_mapper -- [Optional] Selector to retrieve an\n            observable sequence that represents the timeout between the\n            current element and the next element.\n        other -- [Optional] Sequence to return in case of a timeout. If not\n            provided, this is set to reactivex.throw().\n\n    Returns:\n        The source sequence switching to the other sequence in case\n    of a timeout.\n    '
    first_timeout_ = first_timeout or reactivex.never()
    other_ = other or reactivex.throw(Exception('Timeout'))

    def timeout_with_mapper(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            subscription = SerialDisposable()
            timer = SerialDisposable()
            original = SingleAssignmentDisposable()
            subscription.disposable = original
            switched = False
            _id = [0]

            def set_timer(timeout: Observable[Any]) -> None:
                if False:
                    while True:
                        i = 10
                my_id = _id[0]

                def timer_wins():
                    if False:
                        return 10
                    return _id[0] == my_id
                d = SingleAssignmentDisposable()
                timer.disposable = d

                def on_next(x: Any) -> None:
                    if False:
                        print('Hello World!')
                    if timer_wins():
                        subscription.disposable = other_.subscribe(observer, scheduler=scheduler)
                    d.dispose()

                def on_error(e: Exception) -> None:
                    if False:
                        for i in range(10):
                            print('nop')
                    if timer_wins():
                        observer.on_error(e)

                def on_completed() -> None:
                    if False:
                        while True:
                            i = 10
                    if timer_wins():
                        subscription.disposable = other_.subscribe(observer)
                d.disposable = timeout.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
            set_timer(first_timeout_)

            def observer_wins():
                if False:
                    while True:
                        i = 10
                res = not switched
                if res:
                    _id[0] += 1
                return res

            def on_next(x: _T) -> None:
                if False:
                    return 10
                if observer_wins():
                    observer.on_next(x)
                    timeout = None
                    try:
                        timeout = timeout_duration_mapper(x) if timeout_duration_mapper else reactivex.never()
                    except Exception as e:
                        observer.on_error(e)
                        return
                    set_timer(timeout)

            def on_error(error: Exception) -> None:
                if False:
                    return 10
                if observer_wins():
                    observer.on_error(error)

            def on_completed() -> None:
                if False:
                    return 10
                if observer_wins():
                    observer.on_completed()
            original.disposable = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
            return CompositeDisposable(subscription, timer)
        return Observable(subscribe)
    return timeout_with_mapper
__all__ = ['timeout_with_mapper_']