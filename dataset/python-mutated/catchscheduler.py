from datetime import datetime
from typing import Callable, Optional, TypeVar, cast
from reactivex import abc, typing
from reactivex.abc.scheduler import SchedulerBase
from reactivex.disposable import Disposable, SingleAssignmentDisposable
from .periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')

class CatchScheduler(PeriodicScheduler):

    def __init__(self, scheduler: abc.SchedulerBase, handler: Callable[[Exception], bool]) -> None:
        if False:
            while True:
                i = 10
        'Wraps a scheduler, passed as constructor argument, adding exception\n        handling for scheduled actions. The handler should return True to\n        indicate it handled the exception successfully. Falsy return values will\n        be taken to indicate that the exception should be escalated (raised by\n        this scheduler).\n\n        Args:\n            scheduler: The scheduler to be wrapped.\n            handler: Callable to handle exceptions raised by wrapped scheduler.\n        '
        super().__init__()
        self._scheduler: abc.SchedulerBase = scheduler
        self._handler: Callable[[Exception], bool] = handler
        self._recursive_original: Optional[abc.SchedulerBase] = None
        self._recursive_wrapper: Optional['CatchScheduler'] = None

    @property
    def now(self) -> datetime:
        if False:
            i = 10
            return i + 15
        "Represents a notion of time for this scheduler. Tasks being\n        scheduled on a scheduler will adhere to the time denoted by this\n        property.\n\n        Returns:\n             The scheduler's current time, as a datetime instance.\n        "
        return self._scheduler.now

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            return 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        action = self._wrap(action)
        return self._scheduler.schedule(action, state=state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        action = self._wrap(action)
        return self._scheduler.schedule_relative(duetime, action, state=state)

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        action = self._wrap(action)
        return self._scheduler.schedule_absolute(duetime, action, state=state)

    def schedule_periodic(self, period: typing.RelativeTime, action: typing.ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            return 10
        'Schedules a periodic piece of work.\n\n        Args:\n            period: Period in seconds or timedelta for running the\n                work periodically.\n            action: Action to be executed.\n            state: [Optional] Initial state passed to the action upon\n                the first iteration.\n\n        Returns:\n            The disposable object used to cancel the scheduled\n            recurring action (best effort).\n        '
        schedule_periodic = getattr(self._scheduler, 'schedule_periodic')
        if not callable(schedule_periodic):
            raise NotImplementedError
        disp: SingleAssignmentDisposable = SingleAssignmentDisposable()
        failed: bool = False

        def periodic(state: Optional[_TState]=None) -> Optional[_TState]:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal failed
            if failed:
                return None
            try:
                return action(state)
            except Exception as ex:
                failed = True
                if not self._handler(ex):
                    raise
                disp.dispose()
                return None
        scheduler = cast(PeriodicScheduler, self._scheduler)
        disp.disposable = scheduler.schedule_periodic(period, periodic, state=state)
        return disp

    def _clone(self, scheduler: abc.SchedulerBase) -> 'CatchScheduler':
        if False:
            return 10
        return CatchScheduler(scheduler, self._handler)

    def _wrap(self, action: typing.ScheduledAction[_TState]) -> typing.ScheduledAction[_TState]:
        if False:
            for i in range(10):
                print('nop')
        parent = self

        def wrapped_action(self: abc.SchedulerBase, state: Optional[_TState]) -> Optional[abc.DisposableBase]:
            if False:
                while True:
                    i = 10
            try:
                return action(parent._get_recursive_wrapper(self), state)
            except Exception as ex:
                if not parent._handler(ex):
                    raise
                return Disposable()
        return wrapped_action

    def _get_recursive_wrapper(self, scheduler: SchedulerBase) -> 'CatchScheduler':
        if False:
            return 10
        if self._recursive_wrapper is None or self._recursive_original != scheduler:
            self._recursive_original = scheduler
            wrapper = self._clone(scheduler)
            wrapper._recursive_original = scheduler
            wrapper._recursive_wrapper = wrapper
            self._recursive_wrapper = wrapper
        return self._recursive_wrapper