from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy import func, or_, select
from airflow.models.dagrun import DagRun
from airflow.models.taskinstance import PAST_DEPENDS_MET, TaskInstance as TI
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.db import exists_query
from airflow.utils.session import provide_session
from airflow.utils.state import TaskInstanceState
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.models.operator import Operator
_SUCCESSFUL_STATES = (TaskInstanceState.SKIPPED, TaskInstanceState.SUCCESS)

class PrevDagrunDep(BaseTIDep):
    """
    Is the past dagrun in a state that allows this task instance to run.

    For example, did this task instance's task in the previous dagrun complete
    if we are depending on past?
    """
    NAME = 'Previous Dagrun State'
    IGNORABLE = True
    IS_TASK_DEP = True

    @staticmethod
    def _push_past_deps_met_xcom_if_needed(ti: TI, dep_context):
        if False:
            print('Hello World!')
        if dep_context.wait_for_past_depends_before_skipping:
            ti.xcom_push(key=PAST_DEPENDS_MET, value=True)

    @staticmethod
    def _has_tis(dagrun: DagRun, task_id: str, *, session: Session) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if a task has presence in the specified DAG run.\n\n        This function exists for easy mocking in tests.\n        '
        return exists_query(TI.dag_id == dagrun.dag_id, TI.task_id == task_id, TI.run_id == dagrun.run_id, session=session)

    @staticmethod
    def _has_any_prior_tis(ti: TI, *, session: Session) -> bool:
        if False:
            return 10
        'Check if a task has ever been run before.\n\n        This function exists for easy mocking in tests.\n        '
        return exists_query(TI.dag_id == ti.dag_id, TI.task_id == ti.task_id, TI.execution_date < ti.execution_date, session=session)

    @staticmethod
    def _count_unsuccessful_tis(dagrun: DagRun, task_id: str, *, session: Session) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get a count of unsuccessful task instances in a given run.\n\n        Due to historical design considerations, "unsuccessful" here means the\n        task instance is not in either SUCCESS or SKIPPED state. This means that\n        unfinished states such as RUNNING are considered unsuccessful.\n\n        This function exists for easy mocking in tests.\n        '
        return session.scalar(select(func.count()).where(TI.dag_id == dagrun.dag_id, TI.task_id == task_id, TI.run_id == dagrun.run_id, or_(TI.state.is_(None), TI.state.not_in(_SUCCESSFUL_STATES))))

    @staticmethod
    def _has_unsuccessful_dependants(dagrun: DagRun, task: Operator, *, session: Session) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if any of the task\'s dependants are unsuccessful in a given run.\n\n        Due to historical design considerations, "unsuccessful" here means the\n        task instance is not in either SUCCESS or SKIPPED state. This means that\n        unfinished states such as RUNNING are considered unsuccessful.\n\n        This function exists for easy mocking in tests.\n        '
        if not task.downstream_task_ids:
            return False
        return exists_query(TI.dag_id == dagrun.dag_id, TI.task_id.in_(task.downstream_task_ids), TI.run_id == dagrun.run_id, or_(TI.state.is_(None), TI.state.not_in(_SUCCESSFUL_STATES)), session=session)

    @provide_session
    def _get_dep_statuses(self, ti: TI, session: Session, dep_context):
        if False:
            return 10
        if dep_context.ignore_depends_on_past:
            self._push_past_deps_met_xcom_if_needed(ti, dep_context)
            reason = 'The context specified that the state of past DAGs could be ignored.'
            yield self._passing_status(reason=reason)
            return
        if not ti.task.depends_on_past:
            self._push_past_deps_met_xcom_if_needed(ti, dep_context)
            yield self._passing_status(reason='The task did not have depends_on_past set.')
            return
        dr = ti.get_dagrun(session=session)
        if not dr:
            self._push_past_deps_met_xcom_if_needed(ti, dep_context)
            yield self._passing_status(reason='This task instance does not belong to a DAG.')
            return
        catchup = ti.task.dag and ti.task.dag.catchup
        if catchup:
            last_dagrun = DagRun.get_previous_scheduled_dagrun(dr.id, session)
        else:
            last_dagrun = DagRun.get_previous_dagrun(dr, session=session)
        if not last_dagrun:
            self._push_past_deps_met_xcom_if_needed(ti, dep_context)
            yield self._passing_status(reason='This task instance was the first task instance for its task.')
            return
        if catchup and last_dagrun.execution_date < ti.task.start_date:
            self._push_past_deps_met_xcom_if_needed(ti, dep_context)
            yield self._passing_status(reason='This task instance was the first task instance for its task.')
            return
        if not self._has_tis(last_dagrun, ti.task_id, session=session):
            if ti.task.ignore_first_depends_on_past:
                if not self._has_any_prior_tis(ti, session=session):
                    self._push_past_deps_met_xcom_if_needed(ti, dep_context)
                    yield self._passing_status(reason='ignore_first_depends_on_past is true for this task and it is the first task instance for its task.')
                    return
            yield self._failing_status(reason="depends_on_past is true for this task's DAG, but the previous task instance has not run yet.")
            return
        unsuccessful_tis_count = self._count_unsuccessful_tis(last_dagrun, ti.task_id, session=session)
        if unsuccessful_tis_count > 0:
            reason = f'depends_on_past is true for this task, but {unsuccessful_tis_count} previous task instance(s) are not in a successful state.'
            yield self._failing_status(reason=reason)
            return
        if ti.task.wait_for_downstream and self._has_unsuccessful_dependants(last_dagrun, ti.task, session=session):
            yield self._failing_status(reason="The tasks downstream of the previous task instance(s) haven't completed, and wait_for_downstream is True.")
            return
        self._push_past_deps_met_xcom_if_needed(ti, dep_context)