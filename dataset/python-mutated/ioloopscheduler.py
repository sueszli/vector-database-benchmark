import logging
from datetime import datetime
from typing import Any, Optional, TypeVar
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class IOLoopScheduler(PeriodicScheduler):
    """A scheduler that schedules work via the Tornado I/O main event loop.

    Note, as of Tornado 6, this is just a wrapper around the asyncio loop.

    http://tornado.readthedocs.org/en/latest/ioloop.html"""

    def __init__(self, loop: Any) -> None:
        if False:
            print('Hello World!')
        'Create a new IOLoopScheduler.\n\n        Args:\n            loop: The ioloop to use; typically, you would get this by\n                tornado import ioloop; ioloop.IOLoop.current()\n        '
        super().__init__()
        self._loop = loop

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        sad = SingleAssignmentDisposable()
        disposed = False

        def interval() -> None:
            if False:
                print('Hello World!')
            if not disposed:
                sad.disposable = self.invoke_action(action, state=state)
        self._loop.add_callback(interval)

        def dispose() -> None:
            if False:
                print('Hello World!')
            nonlocal disposed
            disposed = True
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        seconds = self.to_seconds(duetime)
        if seconds <= 0.0:
            return self.schedule(action, state=state)
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                while True:
                    i = 10
            sad.disposable = self.invoke_action(action, state=state)
        log.debug('timeout: %s', seconds)
        timer = self._loop.call_later(seconds, interval)

        def dispose() -> None:
            if False:
                i = 10
                return i + 15
            self._loop.remove_timeout(timer)
            self._loop.remove_timeout(timer)
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state=state)

    @property
    def now(self) -> datetime:
        if False:
            print('Hello World!')
        "Represents a notion of time for this scheduler. Tasks being\n        scheduled on a scheduler will adhere to the time denoted by this\n        property.\n\n        Returns:\n             The scheduler's current time, as a datetime instance.\n        "
        return self.to_datetime(self._loop.time())