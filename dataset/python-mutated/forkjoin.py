from typing import Any, List, Optional, Tuple, cast
from reactivex import Observable, abc
from reactivex.disposable import CompositeDisposable, SingleAssignmentDisposable

def fork_join_(*sources: Observable[Any]) -> Observable[Tuple[Any, ...]]:
    if False:
        i = 10
        return i + 15
    'Wait for observables to complete and then combine last values\n    they emitted into a tuple. Whenever any of that observables completes\n    without emitting any value, result sequence will complete at that moment as well.\n\n    Examples:\n        >>> obs = reactivex.fork_join(obs1, obs2, obs3)\n\n    Returns:\n        An observable sequence containing the result of combining last element from\n        each source in given sequence.\n    '
    parent = sources[0]

    def subscribe(observer: abc.ObserverBase[Tuple[Any, ...]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        n = len(sources)
        values = [None] * n
        is_done = [False] * n
        has_value = [False] * n

        def done(i: int) -> None:
            if False:
                i = 10
                return i + 15
            is_done[i] = True
            if not has_value[i]:
                observer.on_completed()
                return
            if all(is_done):
                if all(has_value):
                    observer.on_next(tuple(values))
                    observer.on_completed()
                else:
                    observer.on_completed()
        subscriptions: List[SingleAssignmentDisposable] = [cast(SingleAssignmentDisposable, None)] * n

        def _subscribe(i: int) -> None:
            if False:
                i = 10
                return i + 15
            subscriptions[i] = SingleAssignmentDisposable()

            def on_next(value: Any) -> None:
                if False:
                    print('Hello World!')
                with parent.lock:
                    values[i] = value
                    has_value[i] = True

            def on_completed() -> None:
                if False:
                    print('Hello World!')
                with parent.lock:
                    done(i)
            subscriptions[i].disposable = sources[i].subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
        for i in range(n):
            _subscribe(i)
        return CompositeDisposable(subscriptions)
    return Observable(subscribe)
__all__ = ['fork_join_']