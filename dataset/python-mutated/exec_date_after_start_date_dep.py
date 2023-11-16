from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session

class ExecDateAfterStartDateDep(BaseTIDep):
    """Determines whether a task's execution date is after start date."""
    NAME = 'Execution Date'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            return 10
        if ti.task.start_date and ti.execution_date < ti.task.start_date:
            yield self._failing_status(reason=f"The execution date is {ti.execution_date.isoformat()} but this is before the task's start date {ti.task.start_date.isoformat()}.")
        if ti.task.dag and ti.task.dag.start_date and (ti.execution_date < ti.task.dag.start_date):
            yield self._failing_status(reason=f"The execution date is {ti.execution_date.isoformat()} but this is before the task's DAG's start date {ti.task.dag.start_date.isoformat()}.")