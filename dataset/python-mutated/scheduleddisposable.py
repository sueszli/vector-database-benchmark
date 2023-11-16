from threading import RLock
from typing import Any
from reactivex import abc
from .singleassignmentdisposable import SingleAssignmentDisposable

class ScheduledDisposable(abc.DisposableBase):
    """Represents a disposable resource whose disposal invocation will
    be scheduled on the specified Scheduler"""

    def __init__(self, scheduler: abc.SchedulerBase, disposable: abc.DisposableBase) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new instance of the ScheduledDisposable class\n        that uses a Scheduler on which to dispose the disposable.'
        self.scheduler = scheduler
        self.disposable = SingleAssignmentDisposable()
        self.disposable.disposable = disposable
        self.lock = RLock()
        super().__init__()

    @property
    def is_disposed(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.disposable.is_disposed

    def dispose(self) -> None:
        if False:
            return 10
        'Disposes the wrapped disposable on the provided scheduler.'

        def action(scheduler: abc.SchedulerBase, state: Any) -> None:
            if False:
                i = 10
                return i + 15
            'Scheduled dispose action'
            self.disposable.dispose()
        self.scheduler.schedule(action)