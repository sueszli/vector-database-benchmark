"""An in process scheduler for managing AdaNet phases."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Iterator
from adanet.experimental.schedulers import scheduler
from adanet.experimental.work_units.work_unit import WorkUnit

class InProcessScheduler(scheduler.Scheduler):
    """A scheduler that executes in a single process."""

    def schedule(self, work_units: Iterator[WorkUnit]):
        if False:
            print('Hello World!')
        'Schedules and execute work units in a single process.\n\n    Args:\n      work_units: An iterator that yields `WorkUnit` instances.\n    '
        for work_unit in work_units:
            work_unit.execute()