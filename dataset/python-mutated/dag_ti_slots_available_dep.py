from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session

class DagTISlotsAvailableDep(BaseTIDep):
    """Determines whether a DAG maximum number of running tasks has been reached."""
    NAME = 'Task Instance Slots Available'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            for i in range(10):
                print('nop')
        if ti.task.dag.get_concurrency_reached(session):
            yield self._failing_status(reason=f"The maximum number of running tasks ({ti.task.dag.max_active_tasks}) for this task's DAG '{ti.dag_id}' has been reached.")