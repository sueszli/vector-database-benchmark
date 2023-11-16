from threading import Lock, Timer
from typing import MutableMapping, Optional, TypeVar
from weakref import WeakKeyDictionary
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from .periodicscheduler import PeriodicScheduler
_TState = TypeVar('_TState')

class TimeoutScheduler(PeriodicScheduler):
    """A scheduler that schedules work via a timed callback."""
    _lock = Lock()
    _global: MutableMapping[type, 'TimeoutScheduler'] = WeakKeyDictionary()

    @classmethod
    def singleton(cls) -> 'TimeoutScheduler':
        if False:
            for i in range(10):
                print('nop')
        with TimeoutScheduler._lock:
            try:
                self = TimeoutScheduler._global[cls]
            except KeyError:
                self = super().__new__(cls)
                TimeoutScheduler._global[cls] = self
        return self

    def __new__(cls) -> 'TimeoutScheduler':
        if False:
            while True:
                i = 10
        return cls.singleton()

    def schedule(self, action: abc.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                print('Hello World!')
            sad.disposable = self.invoke_action(action, state)
        timer = Timer(0, interval)
        timer.daemon = True
        timer.start()

        def dispose() -> None:
            if False:
                i = 10
                return i + 15
            timer.cancel()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_relative(self, duetime: typing.RelativeTime, action: abc.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        seconds = self.to_seconds(duetime)
        if seconds <= 0.0:
            return self.schedule(action, state)
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                while True:
                    i = 10
            sad.disposable = self.invoke_action(action, state)
        timer = Timer(seconds, interval)
        timer.daemon = True
        timer.start()

        def dispose() -> None:
            if False:
                i = 10
                return i + 15
            timer.cancel()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: abc.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            return 10
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state)
__all__ = ['TimeoutScheduler']