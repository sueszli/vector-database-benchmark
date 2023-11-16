from __future__ import annotations
from collections import defaultdict
from unittest.mock import MagicMock
from airflow.executors.base_executor import BaseExecutor
from airflow.models.taskinstancekey import TaskInstanceKey
from airflow.utils.session import create_session
from airflow.utils.state import State

class MockExecutor(BaseExecutor):
    """
    TestExecutor is used for unit testing purposes.
    """

    def __init__(self, do_update=True, *args, **kwargs):
        if False:
            print('Hello World!')
        self.do_update = do_update
        self._running = []
        self.callback_sink = MagicMock()
        self.history = []
        self.sorted_tasks = []
        self.mock_task_results = defaultdict(self.success)
        super().__init__(*args, **kwargs)

    def success(self):
        if False:
            return 10
        return State.SUCCESS

    def heartbeat(self):
        if False:
            while True:
                i = 10
        if not self.do_update:
            return
        with create_session() as session:
            self.history.append(list(self.queued_tasks.values()))

            def sort_by(item):
                if False:
                    for i in range(10):
                        print('nop')
                (key, val) = item
                (dag_id, task_id, date, try_number, map_index) = key
                (_, prio, _, _) = val
                return (-prio, date, dag_id, task_id, map_index, try_number)
            open_slots = self.parallelism - len(self.running)
            sorted_queue = sorted(self.queued_tasks.items(), key=sort_by)
            for (key, (_, _, _, ti)) in sorted_queue[:open_slots]:
                self.queued_tasks.pop(key)
                ti._try_number += 1
                state = self.mock_task_results[key]
                ti.set_state(state, session=session)
                self.change_state(key, state)

    def terminate(self):
        if False:
            while True:
                i = 10
        pass

    def end(self):
        if False:
            print('Hello World!')
        self.sync()

    def change_state(self, key, state, info=None):
        if False:
            print('Hello World!')
        super().change_state(key, state, info=info)
        self.sorted_tasks.append((key, (state, info)))

    def mock_task_fail(self, dag_id, task_id, run_id: str, try_number=1):
        if False:
            print('Hello World!')
        '\n        Set the mock outcome of running this particular task instances to\n        FAILED.\n\n        If the task identified by the tuple ``(dag_id, task_id, date,\n        try_number)`` is run by this executor its state will be FAILED.\n        '
        assert isinstance(run_id, str)
        self.mock_task_results[TaskInstanceKey(dag_id, task_id, run_id, try_number)] = State.FAILED