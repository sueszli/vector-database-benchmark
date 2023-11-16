from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils import timezone
from airflow.utils.session import provide_session

class RunnableExecDateDep(BaseTIDep):
    """Determines whether a task's execution date is valid."""
    NAME = 'Execution Date'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            for i in range(10):
                print('nop')
        cur_date = timezone.utcnow()
        logical_date = ti.get_dagrun(session).execution_date
        if logical_date > cur_date and (not ti.task.dag.allow_future_exec_dates):
            yield self._failing_status(reason=f'Execution date {logical_date.isoformat()} is in the future (the current date is {cur_date.isoformat()}).')
        if ti.task.end_date and logical_date > ti.task.end_date:
            yield self._failing_status(reason=f"The execution date is {logical_date.isoformat()} but this is after the task's end date {ti.task.end_date.isoformat()}.")
        if ti.task.dag and ti.task.dag.end_date and (logical_date > ti.task.dag.end_date):
            yield self._failing_status(reason=f"The execution date is {logical_date.isoformat()} but this is after the task's DAG's end date {ti.task.dag.end_date.isoformat()}.")