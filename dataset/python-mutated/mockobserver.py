from typing import List, TypeVar
from reactivex import abc
from reactivex.notification import OnCompleted, OnError, OnNext
from reactivex.scheduler import VirtualTimeScheduler
from .recorded import Recorded
_T = TypeVar('_T')

class MockObserver(abc.ObserverBase[_T]):

    def __init__(self, scheduler: VirtualTimeScheduler) -> None:
        if False:
            return 10
        self.scheduler = scheduler
        self.messages: List[Recorded[_T]] = []

    def on_next(self, value: _T) -> None:
        if False:
            print('Hello World!')
        self.messages.append(Recorded(self.scheduler.clock, OnNext(value)))

    def on_error(self, error: Exception) -> None:
        if False:
            print('Hello World!')
        self.messages.append(Recorded(self.scheduler.clock, OnError(error)))

    def on_completed(self) -> None:
        if False:
            print('Hello World!')
        self.messages.append(Recorded(self.scheduler.clock, OnCompleted()))