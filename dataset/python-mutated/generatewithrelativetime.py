from typing import Any, Callable, Optional, TypeVar, cast
from reactivex import Observable, abc
from reactivex.disposable import MultipleAssignmentDisposable
from reactivex.scheduler import TimeoutScheduler
from reactivex.typing import Mapper, Predicate, RelativeTime
_TState = TypeVar('_TState')

def generate_with_relative_time_(initial_state: _TState, condition: Predicate[_TState], iterate: Mapper[_TState, _TState], time_mapper: Callable[[_TState], RelativeTime]) -> Observable[_TState]:
    if False:
        return 10
    'Generates an observable sequence by iterating a state from an\n    initial state until the condition fails.\n\n    Example:\n        res = source.generate_with_relative_time(\n            0, lambda x: True, lambda x: x + 1, lambda x: 0.5\n        )\n\n    Args:\n        initial_state: Initial state.\n        condition: Condition to terminate generation (upon returning\n            false).\n        iterate: Iteration step function.\n        time_mapper: Time mapper function to control the speed of\n            values being produced each iteration, returning relative\n            times, i.e. either floats denoting seconds or instances of\n            timedelta.\n\n    Returns:\n        The generated sequence.\n    '

    def subscribe(observer: abc.ObserverBase[_TState], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        scheduler = scheduler or TimeoutScheduler.singleton()
        mad = MultipleAssignmentDisposable()
        state = initial_state
        has_result = False
        result: _TState = cast(_TState, None)
        first = True
        time: Optional[RelativeTime] = None

        def action(scheduler: abc.SchedulerBase, _: Any) -> None:
            if False:
                return 10
            nonlocal state
            nonlocal has_result
            nonlocal result
            nonlocal first
            nonlocal time
            if has_result:
                observer.on_next(result)
            try:
                if first:
                    first = False
                else:
                    state = iterate(state)
                has_result = condition(state)
                if has_result:
                    result = state
                    time = time_mapper(state)
            except Exception as e:
                observer.on_error(e)
                return
            if has_result:
                assert time
                mad.disposable = scheduler.schedule_relative(time, action)
            else:
                observer.on_completed()
        mad.disposable = scheduler.schedule_relative(0, action)
        return mad
    return Observable(subscribe)
__all__ = ['generate_with_relative_time_']