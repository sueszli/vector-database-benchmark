from __future__ import annotations
from airflow.executors.executor_loader import ExecutorLoader
from airflow.models.taskreschedule import TaskReschedule
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils import timezone
from airflow.utils.session import provide_session
from airflow.utils.state import TaskInstanceState

class ReadyToRescheduleDep(BaseTIDep):
    """Determines whether a task is ready to be rescheduled."""
    NAME = 'Ready To Reschedule'
    IGNORABLE = True
    IS_TASK_DEP = True
    RESCHEDULEABLE_STATES = {TaskInstanceState.UP_FOR_RESCHEDULE, None}

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            return 10
        "\n        Determine whether a task is ready to be rescheduled.\n\n        Only tasks in NONE state with at least one row in task_reschedule table are\n        handled by this dependency class, otherwise this dependency is considered as passed.\n        This dependency fails if the latest reschedule request's reschedule date is still\n        in the future.\n        "
        from airflow.models.mappedoperator import MappedOperator
        is_mapped = isinstance(ti.task, MappedOperator)
        (executor, _) = ExecutorLoader.import_default_executor_cls()
        if not is_mapped and (not getattr(ti.task, 'reschedule', False)) and (not executor.change_sensor_mode_to_reschedule):
            yield self._passing_status(reason='Task is not in reschedule mode.')
            return
        if dep_context.ignore_in_reschedule_period:
            yield self._passing_status(reason='The context specified that being in a reschedule period was permitted.')
            return
        if ti.state not in self.RESCHEDULEABLE_STATES:
            yield self._passing_status(reason='The task instance is not in State_UP_FOR_RESCHEDULE or NONE state.')
            return
        next_reschedule_date = session.scalar(TaskReschedule.stmt_for_task_instance(ti, descending=True).with_only_columns(TaskReschedule.reschedule_date).limit(1))
        if not next_reschedule_date:
            if is_mapped:
                yield self._passing_status(reason='The task is mapped and not in reschedule mode')
                return
            yield self._passing_status(reason='There is no reschedule request for this task instance.')
            return
        now = timezone.utcnow()
        if now >= next_reschedule_date:
            yield self._passing_status(reason='Task instance id ready for reschedule.')
            return
        yield self._failing_status(reason=f'Task is not ready for reschedule yet but will be rescheduled automatically. Current date is {now.isoformat()} and task will be rescheduled at {next_reschedule_date.isoformat()}.')