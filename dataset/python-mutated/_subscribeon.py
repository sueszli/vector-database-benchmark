from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.disposable import ScheduledDisposable, SerialDisposable, SingleAssignmentDisposable
_T = TypeVar('_T')

def subscribe_on_(scheduler: abc.SchedulerBase) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        i = 10
        return i + 15

    def subscribe_on(source: Observable[_T]) -> Observable[_T]:
        if False:
            i = 10
            return i + 15
        'Subscribe on the specified scheduler.\n\n        Wrap the source sequence in order to run its subscription and\n        unsubscription logic on the specified scheduler. This operation\n        is not commonly used; see the remarks section for more\n        information on the distinction between subscribe_on and\n        observe_on.\n\n        This only performs the side-effects of subscription and\n        unsubscription on the specified scheduler. In order to invoke\n        observer callbacks on a scheduler, use observe_on.\n\n        Args:\n            source: The source observable..\n\n        Returns:\n            The source sequence whose subscriptions and\n            un-subscriptions happen on the specified scheduler.\n        '

        def subscribe(observer: abc.ObserverBase[_T], _: Optional[abc.SchedulerBase]=None):
            if False:
                return 10
            m = SingleAssignmentDisposable()
            d = SerialDisposable()
            d.disposable = m

            def action(scheduler: abc.SchedulerBase, state: Optional[Any]=None):
                if False:
                    print('Hello World!')
                d.disposable = ScheduledDisposable(scheduler, source.subscribe(observer))
            m.disposable = scheduler.schedule(action)
            return d
        return Observable(subscribe)
    return subscribe_on
__all__ = ['subscribe_on_']