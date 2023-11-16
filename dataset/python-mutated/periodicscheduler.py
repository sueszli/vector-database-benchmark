from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar, Union
from .disposable import DisposableBase
from .scheduler import RelativeTime, ScheduledAction
_TState = TypeVar('_TState')
ScheduledPeriodicAction = Callable[[Optional[_TState]], Optional[_TState]]
ScheduledSingleOrPeriodicAction = Union[ScheduledAction[_TState], ScheduledPeriodicAction[_TState]]

class PeriodicSchedulerBase(ABC):
    """PeriodicScheduler abstract base class."""
    __slots__ = ()

    @abstractmethod
    def schedule_periodic(self, period: RelativeTime, action: ScheduledPeriodicAction[_TState], state: Optional[_TState]=None) -> DisposableBase:
        if False:
            return 10
        'Schedules a periodic piece of work.\n\n        Args:\n            period: Period in seconds or timedelta for running the\n                work periodically.\n            action: Action to be executed.\n            state: [Optional] Initial state passed to the action upon\n                the first iteration.\n\n        Returns:\n            The disposable object used to cancel the scheduled\n            recurring action (best effort).\n        '
        return NotImplemented
__all__ = ['PeriodicSchedulerBase', 'ScheduledPeriodicAction', 'ScheduledSingleOrPeriodicAction', 'RelativeTime']