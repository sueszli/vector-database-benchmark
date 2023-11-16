from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attr
from .. import _core
from .._util import final
if TYPE_CHECKING:
    from collections.abc import Iterator
    from ._run import Task

@attr.s(frozen=True, slots=True)
class ParkingLotStatistics:
    """An object containing debugging information for a ParkingLot.

    Currently, the following fields are defined:

    * ``tasks_waiting`` (int): The number of tasks blocked on this lot's
      :meth:`trio.lowlevel.ParkingLot.park` method.

    """
    tasks_waiting: int = attr.ib()

@final
@attr.s(eq=False, hash=False, slots=True)
class ParkingLot:
    """A fair wait queue with cancellation and requeueing.

    This class encapsulates the tricky parts of implementing a wait
    queue. It's useful for implementing higher-level synchronization
    primitives like queues and locks.

    In addition to the methods below, you can use ``len(parking_lot)`` to get
    the number of parked tasks, and ``if parking_lot: ...`` to check whether
    there are any parked tasks.

    """
    _parked: OrderedDict[Task, None] = attr.ib(factory=OrderedDict, init=False)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of parked tasks.'
        return len(self._parked)

    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'True if there are parked tasks, False otherwise.'
        return bool(self._parked)

    @_core.enable_ki_protection
    async def park(self) -> None:
        """Park the current task until woken by a call to :meth:`unpark` or
        :meth:`unpark_all`.

        """
        task = _core.current_task()
        self._parked[task] = None
        task.custom_sleep_data = self

        def abort_fn(_: _core.RaiseCancelT) -> _core.Abort:
            if False:
                i = 10
                return i + 15
            del task.custom_sleep_data._parked[task]
            return _core.Abort.SUCCEEDED
        await _core.wait_task_rescheduled(abort_fn)

    def _pop_several(self, count: int | float) -> Iterator[Task]:
        if False:
            return 10
        if isinstance(count, float):
            if math.isinf(count):
                count = len(self._parked)
            else:
                raise ValueError('Cannot pop a non-integer number of tasks.')
        else:
            count = min(count, len(self._parked))
        for _ in range(count):
            (task, _) = self._parked.popitem(last=False)
            yield task

    @_core.enable_ki_protection
    def unpark(self, *, count: int | float=1) -> list[Task]:
        if False:
            while True:
                i = 10
        'Unpark one or more tasks.\n\n        This wakes up ``count`` tasks that are blocked in :meth:`park`. If\n        there are fewer than ``count`` tasks parked, then wakes as many tasks\n        are available and then returns successfully.\n\n        Args:\n          count (int | math.inf): the number of tasks to unpark.\n\n        '
        tasks = list(self._pop_several(count))
        for task in tasks:
            _core.reschedule(task)
        return tasks

    def unpark_all(self) -> list[Task]:
        if False:
            print('Hello World!')
        'Unpark all parked tasks.'
        return self.unpark(count=len(self))

    @_core.enable_ki_protection
    def repark(self, new_lot: ParkingLot, *, count: int | float=1) -> None:
        if False:
            i = 10
            return i + 15
        'Move parked tasks from one :class:`ParkingLot` object to another.\n\n        This dequeues ``count`` tasks from one lot, and requeues them on\n        another, preserving order. For example::\n\n           async def parker(lot):\n               print("sleeping")\n               await lot.park()\n               print("woken")\n\n           async def main():\n               lot1 = trio.lowlevel.ParkingLot()\n               lot2 = trio.lowlevel.ParkingLot()\n               async with trio.open_nursery() as nursery:\n                   nursery.start_soon(parker, lot1)\n                   await trio.testing.wait_all_tasks_blocked()\n                   assert len(lot1) == 1\n                   assert len(lot2) == 0\n                   lot1.repark(lot2)\n                   assert len(lot1) == 0\n                   assert len(lot2) == 1\n                   # This wakes up the task that was originally parked in lot1\n                   lot2.unpark()\n\n        If there are fewer than ``count`` tasks parked, then reparks as many\n        tasks as are available and then returns successfully.\n\n        Args:\n          new_lot (ParkingLot): the parking lot to move tasks to.\n          count (int|math.inf): the number of tasks to move.\n\n        '
        if not isinstance(new_lot, ParkingLot):
            raise TypeError('new_lot must be a ParkingLot')
        for task in self._pop_several(count):
            new_lot._parked[task] = None
            task.custom_sleep_data = new_lot

    def repark_all(self, new_lot: ParkingLot) -> None:
        if False:
            print('Hello World!')
        'Move all parked tasks from one :class:`ParkingLot` object to\n        another.\n\n        See :meth:`repark` for details.\n\n        '
        return self.repark(new_lot, count=len(self))

    def statistics(self) -> ParkingLotStatistics:
        if False:
            for i in range(10):
                print('nop')
        "Return an object containing debugging information.\n\n        Currently the following fields are defined:\n\n        * ``tasks_waiting``: The number of tasks blocked on this lot's\n          :meth:`park` method.\n\n        "
        return ParkingLotStatistics(tasks_waiting=len(self._parked))