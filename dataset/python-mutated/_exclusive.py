from asyncio import Future
from typing import Callable, Optional, TypeVar, Union
import reactivex
from reactivex import Observable, abc
from reactivex.disposable import CompositeDisposable, SingleAssignmentDisposable
_T = TypeVar('_T')

def exclusive_() -> Callable[[Observable[Observable[_T]]], Observable[_T]]:
    if False:
        return 10
    'Performs a exclusive waiting for the first to finish before\n    subscribing to another observable. Observables that come in between\n    subscriptions will be dropped on the floor.\n\n    Returns:\n        An exclusive observable with only the results that\n        happen when subscribed.\n    '

    def exclusive(source: Observable[Observable[_T]]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                while True:
                    i = 10
            has_current = [False]
            is_stopped = [False]
            m = SingleAssignmentDisposable()
            g = CompositeDisposable()
            g.add(m)

            def on_next(inner_source: Union[Observable[_T], 'Future[_T]']) -> None:
                if False:
                    return 10
                if not has_current[0]:
                    has_current[0] = True
                    inner_source = reactivex.from_future(inner_source) if isinstance(inner_source, Future) else inner_source
                    inner_subscription = SingleAssignmentDisposable()
                    g.add(inner_subscription)

                    def on_completed_inner():
                        if False:
                            return 10
                        g.remove(inner_subscription)
                        has_current[0] = False
                        if is_stopped[0] and len(g) == 1:
                            observer.on_completed()
                    inner_subscription.disposable = inner_source.subscribe(observer.on_next, observer.on_error, on_completed_inner, scheduler=scheduler)

            def on_completed() -> None:
                if False:
                    for i in range(10):
                        print('nop')
                is_stopped[0] = True
                if not has_current[0] and len(g) == 1:
                    observer.on_completed()
            m.disposable = source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
            return g
        return Observable(subscribe)
    return exclusive
__all__ = ['exclusive_']