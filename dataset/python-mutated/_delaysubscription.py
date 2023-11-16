from typing import Any, Callable, Optional, TypeVar
import reactivex
from reactivex import Observable, abc
from reactivex import operators as ops
from reactivex import typing
_T = TypeVar('_T')

def delay_subscription_(duetime: typing.AbsoluteOrRelativeTime, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def delay_subscription(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Time shifts the observable sequence by delaying the subscription.\n\n        Exampeles.\n            >>> res = source.delay_subscription(5)\n\n        Args:\n            source: Source subscription to delay.\n\n        Returns:\n            Time-shifted sequence.\n        '

        def mapper(_: Any) -> Observable[_T]:
            if False:
                print('Hello World!')
            return reactivex.empty()
        return source.pipe(ops.delay_with_mapper(reactivex.timer(duetime, scheduler=scheduler), mapper))
    return delay_subscription
__all__ = ['delay_subscription_']