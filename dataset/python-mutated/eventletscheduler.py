import logging
from datetime import datetime
from typing import Any, Optional, TypeVar
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class EventletScheduler(PeriodicScheduler):
    """A scheduler that schedules work via the eventlet event loop.

    http://eventlet.net/
    """

    def __init__(self, eventlet: Any) -> None:
        if False:
            while True:
                i = 10
        'Create a new EventletScheduler.\n\n        Args:\n            eventlet: The eventlet module to use; typically, you would get this\n                by import eventlet\n        '
        super().__init__()
        self._eventlet = eventlet

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                print('Hello World!')
            sad.disposable = self.invoke_action(action, state=state)
        timer = self._eventlet.spawn(interval)

        def dispose() -> None:
            if False:
                return 10
            timer.kill()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        seconds = self.to_seconds(duetime)
        if seconds <= 0.0:
            return self.schedule(action, state=state)
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                i = 10
                return i + 15
            sad.disposable = self.invoke_action(action, state=state)
        timer = self._eventlet.spawn_after(seconds, interval)

        def dispose() -> None:
            if False:
                while True:
                    i = 10
            timer.kill()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state=state)

    @property
    def now(self) -> datetime:
        if False:
            i = 10
            return i + 15
        "Represents a notion of time for this scheduler. Tasks being\n        scheduled on a scheduler will adhere to the time denoted by this\n        property.\n\n        Returns:\n             The scheduler's current time, as a datetime instance.\n        "
        return self.to_datetime(self._eventlet.hubs.get_hub().clock())