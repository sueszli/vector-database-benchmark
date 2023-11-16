import logging
from datetime import timedelta
from typing import Any, Optional, Set, TypeVar
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger(__name__)

class QtScheduler(PeriodicScheduler):
    """A scheduler for a PyQt5/PySide2 event loop."""

    def __init__(self, qtcore: Any):
        if False:
            while True:
                i = 10
        'Create a new QtScheduler.\n\n        Args:\n            qtcore: The QtCore instance to use; typically you would get this by\n                either import PyQt5.QtCore or import PySide2.QtCore\n        '
        super().__init__()
        self._qtcore = qtcore
        self._periodic_timers: Set[Any] = set()

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self.schedule_relative(0.0, action, state=state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        msecs = max(0, int(self.to_seconds(duetime) * 1000.0))
        sad = SingleAssignmentDisposable()
        is_disposed = False

        def invoke_action() -> None:
            if False:
                return 10
            if not is_disposed:
                sad.disposable = action(self, state)
        log.debug('relative timeout: %sms', msecs)
        self._qtcore.QTimer.singleShot(msecs, invoke_action)

        def dispose() -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal is_disposed
            is_disposed = True
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        delta: timedelta = self.to_datetime(duetime) - self.now
        return self.schedule_relative(delta, action, state=state)

    def schedule_periodic(self, period: typing.RelativeTime, action: typing.ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules a periodic piece of work to be executed in the loop.\n\n        Args:\n             period: Period in seconds for running the work repeatedly.\n             action: Action to be executed.\n             state: [Optional] state to be given to the action function.\n\n         Returns:\n             The disposable object used to cancel the scheduled action\n             (best effort).\n        '
        msecs = max(0, int(self.to_seconds(period) * 1000.0))
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                print('Hello World!')
            nonlocal state
            state = action(state)
        log.debug('periodic timeout: %sms', msecs)
        timer = self._qtcore.QTimer()
        timer.setSingleShot(not period)
        timer.timeout.connect(interval)
        timer.setInterval(msecs)
        self._periodic_timers.add(timer)
        timer.start()

        def dispose() -> None:
            if False:
                i = 10
                return i + 15
            timer.stop()
            self._periodic_timers.remove(timer)
            timer.deleteLater()
        return CompositeDisposable(sad, Disposable(dispose))