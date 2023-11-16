from typing import Callable, Optional, TypeVar
from reactivex import ConnectableObservable, Observable, abc
from reactivex.disposable import Disposable
_T = TypeVar('_T')

def ref_count_() -> Callable[[ConnectableObservable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15
    'Returns an observable sequence that stays connected to the\n    source as long as there is at least one subscription to the\n    observable sequence.\n    '
    connectable_subscription: Optional[abc.DisposableBase] = None
    count = 0

    def ref_count(source: ConnectableObservable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                while True:
                    i = 10
            nonlocal connectable_subscription, count
            count += 1
            should_connect = count == 1
            subscription = source.subscribe(observer, scheduler=scheduler)
            if should_connect:
                connectable_subscription = source.connect(scheduler)

            def dispose() -> None:
                if False:
                    i = 10
                    return i + 15
                nonlocal connectable_subscription, count
                subscription.dispose()
                count -= 1
                if not count and connectable_subscription:
                    connectable_subscription.dispose()
            return Disposable(dispose)
        return Observable(subscribe)
    return ref_count
__all__ = ['ref_count_']