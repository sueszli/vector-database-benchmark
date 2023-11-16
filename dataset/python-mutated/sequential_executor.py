"""
SequentialExecutor.

.. seealso::
    For more information on how the SequentialExecutor works, take a look at the guide:
    :ref:`executor:SequentialExecutor`
"""
from __future__ import annotations
import subprocess
from typing import TYPE_CHECKING, Any
from airflow.executors.base_executor import BaseExecutor
from airflow.utils.state import TaskInstanceState
if TYPE_CHECKING:
    from airflow.executors.base_executor import CommandType
    from airflow.models.taskinstancekey import TaskInstanceKey

class SequentialExecutor(BaseExecutor):
    """
    This executor will only run one task instance at a time.

    It can be used for debugging. It is also the only executor
    that can be used with sqlite since sqlite doesn't support
    multiple connections.

    Since we want airflow to work out of the box, it defaults to this
    SequentialExecutor alongside sqlite as you first install it.
    """
    supports_pickling: bool = False
    is_local: bool = True
    is_single_threaded: bool = True
    is_production: bool = False
    serve_logs: bool = True

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.commands_to_run = []

    def execute_async(self, key: TaskInstanceKey, command: CommandType, queue: str | None=None, executor_config: Any | None=None) -> None:
        if False:
            return 10
        self.validate_airflow_tasks_run_command(command)
        self.commands_to_run.append((key, command))

    def sync(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (key, command) in self.commands_to_run:
            self.log.info('Executing command: %s', command)
            try:
                subprocess.check_call(command, close_fds=True)
                self.change_state(key, TaskInstanceState.SUCCESS)
            except subprocess.CalledProcessError as e:
                self.change_state(key, TaskInstanceState.FAILED)
                self.log.error('Failed to execute task %s.', e)
        self.commands_to_run = []

    def end(self):
        if False:
            for i in range(10):
                print('nop')
        'End the executor.'
        self.heartbeat()

    def terminate(self):
        if False:
            i = 10
            return i + 15
        'Terminate the executor is not doing anything.'