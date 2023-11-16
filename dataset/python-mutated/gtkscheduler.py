from typing import Any, Optional, TypeVar, cast
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from ..periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')

class GtkScheduler(PeriodicScheduler):
    """A scheduler that schedules work via the GLib main loop
    used in GTK+ applications.

    See https://wiki.gnome.org/Projects/PyGObject
    """

    def __init__(self, glib: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Create a new GtkScheduler.\n\n        Args:\n            glib: The GLib module to use; typically, you would get this by\n                >>> import gi\n                >>> gi.require_version('Gtk', '3.0')\n                >>> from gi.repository import GLib\n        "
        super().__init__()
        self._glib = glib

    def _gtk_schedule(self, time: typing.AbsoluteOrRelativeTime, action: typing.ScheduledSingleOrPeriodicAction[_TState], state: Optional[_TState]=None, periodic: bool=False) -> abc.DisposableBase:
        if False:
            return 10
        msecs = max(0, int(self.to_seconds(time) * 1000.0))
        sad = SingleAssignmentDisposable()
        stopped = False

        def timer_handler(_: Any) -> bool:
            if False:
                while True:
                    i = 10
            if stopped:
                return False
            nonlocal state
            if periodic:
                state = cast(typing.ScheduledPeriodicAction[_TState], action)(state)
            else:
                sad.disposable = self.invoke_action(cast(typing.ScheduledAction[_TState], action), state=state)
            return periodic
        self._glib.timeout_add(msecs, timer_handler, None)

        def dispose() -> None:
            if False:
                while True:
                    i = 10
            nonlocal stopped
            stopped = True
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            return 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self._gtk_schedule(0.0, action, state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self._gtk_schedule(duetime, action, state=state)

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self._gtk_schedule(duetime - self.now, action, state=state)

    def schedule_periodic(self, period: typing.RelativeTime, action: typing.ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules a periodic piece of work to be executed in the loop.\n\n        Args:\n             period: Period in seconds for running the work repeatedly.\n             action: Action to be executed.\n             state: [Optional] state to be given to the action function.\n\n         Returns:\n             The disposable object used to cancel the scheduled action\n             (best effort).\n        '
        return self._gtk_schedule(period, action, state=state, periodic=True)