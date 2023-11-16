import math
import types
from typing import Any, Generic, TypeVar, Union
from reactivex import typing
from reactivex.notification import OnCompleted, OnError, OnNext
from .recorded import Recorded
from .subscription import Subscription
_T = TypeVar('_T')

def is_prime(i: int) -> bool:
    if False:
        while True:
            i = 10
    'Tests if number is prime or not'
    if i <= 1:
        return False
    _max = int(math.floor(math.sqrt(i)))
    for j in range(2, _max + 1):
        if not i % j:
            return False
    return True

class OnNextPredicate(Generic[_T]):

    def __init__(self, predicate: typing.Predicate[_T]) -> None:
        if False:
            print('Hello World!')
        self.predicate = predicate

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if other == self:
            return True
        if other is None:
            return False
        if other.kind != 'N':
            return False
        return self.predicate(other.value)

class OnErrorPredicate(Generic[_T]):

    def __init__(self, predicate: typing.Predicate[_T]):
        if False:
            while True:
                i = 10
        self.predicate = predicate

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if other == self:
            return True
        if other is None:
            return False
        if other.kind != 'E':
            return False
        return self.predicate(other.exception)

class ReactiveTest:
    created = 100
    subscribed = 200
    disposed = 1000

    @staticmethod
    def on_next(ticks: int, value: _T) -> Recorded[_T]:
        if False:
            return 10
        if isinstance(value, types.FunctionType):
            return Recorded(ticks, OnNextPredicate(value))
        return Recorded(ticks, OnNext(value))

    @staticmethod
    def on_error(ticks: int, error: Union[Exception, str]) -> Recorded[Any]:
        if False:
            i = 10
            return i + 15
        if isinstance(error, types.FunctionType):
            return Recorded(ticks, OnErrorPredicate(error))
        return Recorded(ticks, OnError(error))

    @staticmethod
    def on_completed(ticks: int) -> Recorded[Any]:
        if False:
            print('Hello World!')
        return Recorded(ticks, OnCompleted())

    @staticmethod
    def subscribe(start: int, end: int) -> Subscription:
        if False:
            while True:
                i = 10
        return Subscription(start, end)