import logging
import threading
from datetime import datetime
from typing import Optional, TypeVar
from reactivex import abc, typing
from reactivex.disposable import Disposable
from reactivex.internal.concurrency import default_thread_factory
from .eventloopscheduler import EventLoopScheduler
from .periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class NewThreadScheduler(PeriodicScheduler):
    """Creates an object that schedules each unit of work on a separate thread."""

    def __init__(self, thread_factory: Optional[typing.StartableFactory]=None) -> None:
        if False:
            return 10
        super().__init__()
        self.thread_factory: typing.StartableFactory = thread_factory or default_thread_factory

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        scheduler = EventLoopScheduler(thread_factory=self.thread_factory, exit_if_empty=True)
        return scheduler.schedule(action, state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        scheduler = EventLoopScheduler(thread_factory=self.thread_factory, exit_if_empty=True)
        return scheduler.schedule_relative(duetime, action, state)

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        dt = self.to_datetime(duetime)
        return self.schedule_relative(dt - self.now, action, state=state)

    def schedule_periodic(self, period: typing.RelativeTime, action: typing.ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules a periodic piece of work.\n\n        Args:\n            period: Period in seconds or timedelta for running the\n                work periodically.\n            action: Action to be executed.\n            state: [Optional] Initial state passed to the action upon\n                the first iteration.\n\n        Returns:\n            The disposable object used to cancel the scheduled\n            recurring action (best effort).\n        '
        seconds: float = self.to_seconds(period)
        timeout: float = seconds
        disposed: threading.Event = threading.Event()

        def run() -> None:
            if False:
                i = 10
                return i + 15
            nonlocal state, timeout
            while True:
                if timeout > 0.0:
                    disposed.wait(timeout)
                if disposed.is_set():
                    return
                time: datetime = self.now
                state = action(state)
                timeout = seconds - (self.now - time).total_seconds()
        thread = self.thread_factory(run)
        thread.start()

        def dispose() -> None:
            if False:
                for i in range(10):
                    print('nop')
            disposed.set()
        return Disposable(dispose)