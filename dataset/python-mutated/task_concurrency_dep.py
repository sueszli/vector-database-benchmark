from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session

class TaskConcurrencyDep(BaseTIDep):
    """This restricts the number of running task instances for a particular task."""
    NAME = 'Task Concurrency'
    IGNORABLE = True
    IS_TASK_DEP = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            i = 10
            return i + 15
        if ti.task.max_active_tis_per_dag is None and ti.task.max_active_tis_per_dagrun is None:
            yield self._passing_status(reason='Task concurrency is not set.')
            return
        if ti.task.max_active_tis_per_dag is not None and ti.get_num_running_task_instances(session) >= ti.task.max_active_tis_per_dag:
            yield self._failing_status(reason='The max task concurrency has been reached.')
            return
        if ti.task.max_active_tis_per_dagrun is not None and ti.get_num_running_task_instances(session, same_dagrun=True) >= ti.task.max_active_tis_per_dagrun:
            yield self._failing_status(reason='The max task concurrency per run has been reached.')
            return
        yield self._passing_status(reason='The max task concurrency has not been reached.')
        return