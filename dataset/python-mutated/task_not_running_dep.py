"""Contains the TaskNotRunningDep."""
from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session
from airflow.utils.state import TaskInstanceState

class TaskNotRunningDep(BaseTIDep):
    """Ensures that the task instance's state is not running."""
    NAME = 'Task Instance Not Running'
    IGNORABLE = False

    def __eq__(self, other):
        if False:
            return 10
        return type(self) == type(other)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(type(self))

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context=None):
        if False:
            print('Hello World!')
        if ti.state != TaskInstanceState.RUNNING:
            yield self._passing_status(reason='Task is not in running state.')
            return
        yield self._failing_status(reason='Task is in the running state')