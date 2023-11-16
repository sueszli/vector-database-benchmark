import logging
from datetime import datetime
from typing import Any, Optional, TypeVar
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class TwistedScheduler(PeriodicScheduler):
    """A scheduler that schedules work via the Twisted reactor mainloop."""

    def __init__(self, reactor: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new TwistedScheduler.\n\n        Args:\n            reactor: The reactor to use; typically, you would get this\n                by from twisted.internet import reactor\n        '
        super().__init__()
        self._reactor = reactor

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self.schedule_relative(0.0, action, state=state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        seconds = max(0.0, self.to_seconds(duetime))
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                return 10
            sad.disposable = action(self, state)
        log.debug('timeout: %s', seconds)
        timer = self._reactor.callLater(seconds, interval)

        def dispose() -> None:
            if False:
                for i in range(10):
                    print('nop')
            if not timer.called:
                timer.cancel()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            return 10
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state=state)

    @property
    def now(self) -> datetime:
        if False:
            while True:
                i = 10
        "Represents a notion of time for this scheduler. Tasks being\n        scheduled on a scheduler will adhere to the time denoted by this\n        property.\n\n        Returns:\n             The scheduler's current time, as a datetime instance.\n        "
        return self.to_datetime(float(self._reactor.seconds()))