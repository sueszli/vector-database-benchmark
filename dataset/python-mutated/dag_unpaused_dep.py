from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session

class DagUnpausedDep(BaseTIDep):
    """Determines whether a task's DAG is not paused."""
    NAME = 'Dag Not Paused'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            print('Hello World!')
        if ti.task.dag.get_is_paused(session):
            yield self._failing_status(reason=f"Task's DAG '{ti.dag_id}' is paused.")