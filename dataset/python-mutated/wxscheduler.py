import logging
from typing import Any, Optional, Set, TypeVar, cast
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class WxScheduler(PeriodicScheduler):
    """A scheduler for a wxPython event loop."""

    def __init__(self, wx: Any) -> None:
        if False:
            while True:
                i = 10
        'Create a new WxScheduler.\n\n        Args:\n            wx: The wx module to use; typically, you would get this by\n                import wx\n        '
        super().__init__()
        self._wx = wx
        timer_class: Any = self._wx.Timer

        class Timer(timer_class):

            def __init__(self, callback: typing.Action) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.callback = callback

            def Notify(self) -> None:
                if False:
                    print('Hello World!')
                self.callback()
        self._timer_class = Timer
        self._timers: Set[Timer] = set()

    def cancel_all(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Cancel all scheduled actions.\n\n        Should be called when destroying wx controls to prevent\n        accessing dead wx objects in actions that might be pending.\n        '
        for timer in self._timers:
            timer.Stop()

    def _wxtimer_schedule(self, time: typing.AbsoluteOrRelativeTime, action: typing.ScheduledSingleOrPeriodicAction[_TState], state: Optional[_TState]=None, periodic: bool=False) -> abc.DisposableBase:
        if False:
            return 10
        scheduler = self
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal state
            if periodic:
                state = cast(typing.ScheduledPeriodicAction[_TState], action)(state)
            else:
                sad.disposable = cast(typing.ScheduledAction[_TState], action)(scheduler, state)
        msecs = max(1, int(self.to_seconds(time) * 1000.0))
        log.debug('timeout wx: %s', msecs)
        timer = self._timer_class(interval)
        if self._wx.IsMainThread():
            timer.Start(msecs, oneShot=not periodic)
        else:
            self._wx.CallAfter(timer.Start, msecs, oneShot=not periodic)
        self._timers.add(timer)

        def dispose() -> None:
            if False:
                for i in range(10):
                    print('nop')
            timer.Stop()
            self._timers.remove(timer)
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        sad = SingleAssignmentDisposable()
        is_disposed = False

        def invoke_action() -> None:
            if False:
                print('Hello World!')
            if not is_disposed:
                sad.disposable = action(self, state)
        self._wx.CallAfter(invoke_action)

        def dispose() -> None:
            if False:
                i = 10
                return i + 15
            nonlocal is_disposed
            is_disposed = True
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self._wxtimer_schedule(duetime, action, state=state)

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self._wxtimer_schedule(duetime - self.now, action, state=state)

    def schedule_periodic(self, period: typing.RelativeTime, action: typing.ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules a periodic piece of work to be executed in the loop.\n\n        Args:\n             period: Period in seconds for running the work repeatedly.\n             action: Action to be executed.\n             state: [Optional] state to be given to the action function.\n\n         Returns:\n             The disposable object used to cancel the scheduled action\n             (best effort).\n        '
        return self._wxtimer_schedule(period, action, state=state, periodic=True)