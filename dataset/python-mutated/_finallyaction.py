from typing import Callable, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import Disposable
_T = TypeVar('_T')

def finally_action_(action: typing.Action) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10

    def finally_action(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        'Invokes a specified action after the source observable\n        sequence terminates gracefully or exceptionally.\n\n        Example:\n            res = finally(source)\n\n        Args:\n            source: Observable sequence.\n\n        Returns:\n            An observable sequence with the action-invoking termination\n            behavior applied.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10
            try:
                subscription = source.subscribe(observer, scheduler=scheduler)
            except Exception:
                action()
                raise

            def dispose():
                if False:
                    i = 10
                    return i + 15
                try:
                    subscription.dispose()
                finally:
                    action()
            return Disposable(dispose)
        return Observable(subscribe)
    return finally_action
__all__ = ['finally_action_']