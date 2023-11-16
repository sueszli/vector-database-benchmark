from typing import Any, Callable, List, Optional, TypeVar, cast
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable, SerialDisposable, SingleAssignmentDisposable
from reactivex.scheduler import TimeoutScheduler
_T = TypeVar('_T')

def debounce_(duetime: typing.RelativeTime, scheduler: Optional[abc.SchedulerBase]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def debounce(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Ignores values from an observable sequence which are followed by\n        another value before duetime.\n\n        Example:\n            >>> res = debounce(source)\n\n        Args:\n            source: Source observable to debounce.\n\n        Returns:\n            An operator function that takes the source observable and\n            returns the debounced observable sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()
            cancelable = SerialDisposable()
            has_value = [False]
            value: List[_T] = [cast(_T, None)]
            _id: List[int] = [0]

            def on_next(x: _T) -> None:
                if False:
                    print('Hello World!')
                has_value[0] = True
                value[0] = x
                _id[0] += 1
                current_id = _id[0]
                d = SingleAssignmentDisposable()
                cancelable.disposable = d

                def action(scheduler: abc.SchedulerBase, state: Any=None) -> None:
                    if False:
                        for i in range(10):
                            print('nop')
                    if has_value[0] and _id[0] == current_id:
                        observer.on_next(value[0])
                    has_value[0] = False
                d.disposable = _scheduler.schedule_relative(duetime, action)

            def on_error(exception: Exception) -> None:
                if False:
                    while True:
                        i = 10
                cancelable.dispose()
                observer.on_error(exception)
                has_value[0] = False
                _id[0] += 1

            def on_completed() -> None:
                if False:
                    i = 10
                    return i + 15
                cancelable.dispose()
                if has_value[0]:
                    observer.on_next(value[0])
                observer.on_completed()
                has_value[0] = False
                _id[0] += 1
            subscription = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler_)
            return CompositeDisposable(subscription, cancelable)
        return Observable(subscribe)
    return debounce

def throttle_with_mapper_(throttle_duration_mapper: Callable[[Any], Observable[Any]]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        for i in range(10):
            print('nop')

    def throttle_with_mapper(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        'Partially applied throttle_with_mapper operator.\n\n        Ignores values from an observable sequence which are followed by\n        another value within a computed throttle duration.\n\n        Example:\n            >>> obs = throttle_with_mapper(source)\n\n        Args:\n            source: The observable source to throttle.\n\n        Returns:\n            The throttled observable sequence.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                i = 10
                return i + 15
            cancelable = SerialDisposable()
            has_value: bool = False
            value: _T = cast(_T, None)
            _id = [0]

            def on_next(x: _T) -> None:
                if False:
                    i = 10
                    return i + 15
                nonlocal value, has_value
                throttle = None
                try:
                    throttle = throttle_duration_mapper(x)
                except Exception as e:
                    observer.on_error(e)
                    return
                has_value = True
                value = x
                _id[0] += 1
                current_id = _id[0]
                d = SingleAssignmentDisposable()
                cancelable.disposable = d

                def on_next(x: Any) -> None:
                    if False:
                        print('Hello World!')
                    nonlocal has_value
                    if has_value and _id[0] == current_id:
                        observer.on_next(value)
                    has_value = False
                    d.dispose()

                def on_completed() -> None:
                    if False:
                        print('Hello World!')
                    nonlocal has_value
                    if has_value and _id[0] == current_id:
                        observer.on_next(value)
                    has_value = False
                    d.dispose()
                d.disposable = throttle.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)

            def on_error(e: Exception) -> None:
                if False:
                    print('Hello World!')
                nonlocal has_value
                cancelable.dispose()
                observer.on_error(e)
                has_value = False
                _id[0] += 1

            def on_completed() -> None:
                if False:
                    while True:
                        i = 10
                nonlocal has_value
                cancelable.dispose()
                if has_value:
                    observer.on_next(value)
                observer.on_completed()
                has_value = False
                _id[0] += 1
            subscription = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
            return CompositeDisposable(subscription, cancelable)
        return Observable(subscribe)
    return throttle_with_mapper
__all__ = ['debounce_', 'throttle_with_mapper_']