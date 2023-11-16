from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils import timezone
from airflow.utils.session import provide_session
from airflow.utils.state import TaskInstanceState

class NotInRetryPeriodDep(BaseTIDep):
    """Determines whether a task is not in retry period."""
    NAME = 'Not In Retry Period'
    IGNORABLE = True
    IS_TASK_DEP = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            for i in range(10):
                print('nop')
        if dep_context.ignore_in_retry_period:
            yield self._passing_status(reason='The context specified that being in a retry period was permitted.')
            return
        if ti.state != TaskInstanceState.UP_FOR_RETRY:
            yield self._passing_status(reason='The task instance was not marked for retrying.')
            return
        cur_date = timezone.utcnow()
        next_task_retry_date = ti.next_retry_datetime()
        if ti.is_premature:
            yield self._failing_status(reason=f'Task is not ready for retry yet but will be retried automatically. Current date is {cur_date.isoformat()} and task will be retried at {next_task_retry_date.isoformat()}.')