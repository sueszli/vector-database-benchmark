import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar
from reactivex import abc, typing
from reactivex.abc.scheduler import AbsoluteTime
from reactivex.internal import ArgumentOutOfRangeException, PriorityQueue
from .periodicscheduler import PeriodicScheduler
from .scheduleditem import ScheduledItem
log = logging.getLogger('Rx')
MAX_SPINNING = 100
_TState = TypeVar('_TState')

class VirtualTimeScheduler(PeriodicScheduler):
    """Virtual Scheduler. This scheduler should work with either
    datetime/timespan or ticks as int/int"""

    def __init__(self, initial_clock: typing.AbsoluteTime=0) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new virtual time scheduler with the specified\n        initial clock value.\n\n        Args:\n            initial_clock: Initial value for the clock.\n        '
        super().__init__()
        self._clock = initial_clock
        self._is_enabled = False
        self._lock: threading.Lock = threading.Lock()
        self._queue: PriorityQueue[ScheduledItem] = PriorityQueue()

    def _get_clock(self) -> AbsoluteTime:
        if False:
            print('Hello World!')
        with self._lock:
            return self._clock
    clock = property(fget=_get_clock)

    @property
    def now(self) -> datetime:
        if False:
            return 10
        "Represents a notion of time for this scheduler. Tasks being\n        scheduled on a scheduler will adhere to the time denoted by this\n        property.\n\n        Returns:\n             The scheduler's current time, as a datetime instance.\n        "
        return self.to_datetime(self._clock)

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        return self.schedule_absolute(self._clock, action, state=state)

    def schedule_relative(self, duetime: typing.RelativeTime, action: abc.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        time: typing.AbsoluteTime = self.add(self._clock, duetime)
        return self.schedule_absolute(time, action, state=state)

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: abc.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        dt = self.to_datetime(duetime)
        si: ScheduledItem = ScheduledItem(self, state, action, dt)
        with self._lock:
            self._queue.enqueue(si)
        return si.disposable

    def start(self) -> Any:
        if False:
            print('Hello World!')
        'Starts the virtual time scheduler.'
        with self._lock:
            if self._is_enabled:
                return
            self._is_enabled = True
        spinning: int = 0
        while True:
            with self._lock:
                if not self._is_enabled or not self._queue:
                    break
                item: ScheduledItem = self._queue.dequeue()
                if item.duetime > self.now:
                    if isinstance(self._clock, datetime):
                        self._clock = item.duetime
                    else:
                        self._clock = self.to_seconds(item.duetime)
                    spinning = 0
                elif spinning > MAX_SPINNING:
                    if isinstance(self._clock, datetime):
                        self.clock += timedelta(microseconds=1000)
                    else:
                        self._clock += 1.0
                    spinning = 0
            if not item.is_cancelled():
                item.invoke()
            spinning += 1
        self.stop()

    def stop(self) -> None:
        if False:
            return 10
        'Stops the virtual time scheduler.'
        with self._lock:
            self._is_enabled = False

    def advance_to(self, time: typing.AbsoluteTime) -> None:
        if False:
            i = 10
            return i + 15
        'Advances the schedulers clock to the specified absolute time,\n        running all work til that point.\n\n        Args:\n            time: Absolute time to advance the schedulers clock to.\n        '
        dt: datetime = self.to_datetime(time)
        with self._lock:
            if self.now > dt:
                raise ArgumentOutOfRangeException()
            if self.now == dt or self._is_enabled:
                return
            self._is_enabled = True
        while True:
            with self._lock:
                if not self._is_enabled or not self._queue:
                    break
                item: ScheduledItem = self._queue.peek()
                if item.duetime > dt:
                    break
                if item.duetime > self.now:
                    if isinstance(self._clock, datetime):
                        self._clock = item.duetime
                    else:
                        self._clock = self.to_seconds(item.duetime)
                self._queue.dequeue()
            if not item.is_cancelled():
                item.invoke()
        with self._lock:
            self._is_enabled = False
            if isinstance(self._clock, datetime):
                self._clock = dt
            else:
                self._clock = self.to_seconds(dt)

    def advance_by(self, time: typing.RelativeTime) -> None:
        if False:
            while True:
                i = 10
        'Advances the schedulers clock by the specified relative time,\n        running all work scheduled for that timespan.\n\n        Args:\n            time: Relative time to advance the schedulers clock by.\n        '
        log.debug('VirtualTimeScheduler.advance_by(time=%s)', time)
        self.advance_to(self.add(self.now, self.to_timedelta(time)))

    def sleep(self, time: typing.RelativeTime) -> None:
        if False:
            print('Hello World!')
        'Advances the schedulers clock by the specified relative time.\n\n        Args:\n            time: Relative time to advance the schedulers clock by.\n        '
        absolute = self.add(self.now, self.to_timedelta(time))
        dt: datetime = self.to_datetime(absolute)
        if self.now > dt:
            raise ArgumentOutOfRangeException()
        with self._lock:
            if isinstance(self._clock, datetime):
                self._clock = dt
            else:
                self._clock = self.to_seconds(dt)

    @classmethod
    def add(cls, absolute: typing.AbsoluteTime, relative: typing.RelativeTime) -> typing.AbsoluteTime:
        if False:
            i = 10
            return i + 15
        'Adds a relative time value to an absolute time value.\n\n        Args:\n            absolute: Absolute virtual time value.\n            relative: Relative virtual time value to add.\n\n        Returns:\n            The resulting absolute virtual time sum value.\n        '
        return cls.to_datetime(absolute) + cls.to_timedelta(relative)