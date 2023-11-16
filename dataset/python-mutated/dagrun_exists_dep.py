from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session
from airflow.utils.state import DagRunState

class DagrunRunningDep(BaseTIDep):
    """Determines whether a task's DagRun is in valid state."""
    NAME = 'Dagrun Running'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            while True:
                i = 10
        dr = ti.get_dagrun(session)
        if dr.state != DagRunState.RUNNING:
            yield self._failing_status(reason=f"Task instance's dagrun was not in the 'running' state but in the state '{dr.state}'.")