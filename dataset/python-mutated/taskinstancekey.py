from __future__ import annotations
from typing import NamedTuple

class TaskInstanceKey(NamedTuple):
    """Key used to identify task instance."""
    dag_id: str
    task_id: str
    run_id: str
    try_number: int = 1
    map_index: int = -1

    @property
    def primary(self) -> tuple[str, str, str, int]:
        if False:
            while True:
                i = 10
        'Return task instance primary key part of the key.'
        return (self.dag_id, self.task_id, self.run_id, self.map_index)

    @property
    def reduced(self) -> TaskInstanceKey:
        if False:
            for i in range(10):
                print('nop')
        'Remake the key by subtracting 1 from try number to match in memory information.'
        return TaskInstanceKey(self.dag_id, self.task_id, self.run_id, max(1, self.try_number - 1), self.map_index)

    def with_try_number(self, try_number: int) -> TaskInstanceKey:
        if False:
            for i in range(10):
                print('nop')
        'Return TaskInstanceKey with provided ``try_number``.'
        return TaskInstanceKey(self.dag_id, self.task_id, self.run_id, try_number, self.map_index)

    @property
    def key(self) -> TaskInstanceKey:
        if False:
            for i in range(10):
                print('nop')
        'For API-compatibly with TaskInstance.\n\n        Returns self\n        '
        return self