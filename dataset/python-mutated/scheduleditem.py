from datetime import datetime
from typing import Any, Optional, TypeVar
from reactivex import abc
from reactivex.disposable import SingleAssignmentDisposable
from .scheduler import Scheduler
_TState = TypeVar('_TState')

class ScheduledItem(object):

    def __init__(self, scheduler: Scheduler, state: Optional[_TState], action: abc.ScheduledAction[_TState], duetime: datetime) -> None:
        if False:
            print('Hello World!')
        self.scheduler: Scheduler = scheduler
        self.state: Optional[Any] = state
        self.action: abc.ScheduledAction[_TState] = action
        self.duetime: datetime = duetime
        self.disposable: SingleAssignmentDisposable = SingleAssignmentDisposable()

    def invoke(self) -> None:
        if False:
            return 10
        ret = self.scheduler.invoke_action(self.action, state=self.state)
        self.disposable.disposable = ret

    def cancel(self) -> None:
        if False:
            return 10
        'Cancels the work item by disposing the resource returned by\n        invoke_core as soon as possible.'
        self.disposable.dispose()

    def is_cancelled(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.disposable.is_disposed

    def __lt__(self, other: 'ScheduledItem') -> bool:
        if False:
            print('Hello World!')
        return self.duetime < other.duetime

    def __gt__(self, other: 'ScheduledItem') -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.duetime > other.duetime

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        try:
            return self.duetime == other.duetime
        except AttributeError:
            return NotImplemented