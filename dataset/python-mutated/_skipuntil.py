from asyncio import Future
from typing import Any, Callable, Optional, TypeVar, Union
from reactivex import Observable, abc, from_future
from reactivex.disposable import CompositeDisposable, SingleAssignmentDisposable
_T = TypeVar('_T')

def skip_until_(other: Union[Observable[Any], 'Future[Any]']) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the values from the source observable sequence only after\n    the other observable sequence produces a value.\n\n    Args:\n        other: The observable sequence that triggers propagation of\n            elements of the source sequence.\n\n    Returns:\n        An observable sequence containing the elements of the source\n    sequence starting from the point the other sequence triggered\n    propagation.\n    '
    if isinstance(other, Future):
        obs: Observable[Any] = from_future(other)
    else:
        obs = other

    def skip_until(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                i = 10
                return i + 15
            is_open = [False]

            def on_next(left: _T) -> None:
                if False:
                    return 10
                if is_open[0]:
                    observer.on_next(left)

            def on_completed() -> None:
                if False:
                    print('Hello World!')
                if is_open[0]:
                    observer.on_completed()
            subs = source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
            subscriptions = CompositeDisposable(subs)
            right_subscription = SingleAssignmentDisposable()
            subscriptions.add(right_subscription)

            def on_next2(x: Any) -> None:
                if False:
                    while True:
                        i = 10
                is_open[0] = True
                right_subscription.dispose()

            def on_completed2():
                if False:
                    return 10
                right_subscription.dispose()
            right_subscription.disposable = obs.subscribe(on_next2, observer.on_error, on_completed2, scheduler=scheduler)
            return subscriptions
        return Observable(subscribe)
    return skip_until
__all__ = ['skip_until_']