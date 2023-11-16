from __future__ import annotations
from airflow.models.taskinstance import PAST_DEPENDS_MET
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep

class NotPreviouslySkippedDep(BaseTIDep):
    """
    Determine if this task should be skipped.

    Based on any of the task's direct upstream relatives have decided this task should
    be skipped.
    """
    NAME = 'Not Previously Skipped'
    IGNORABLE = True
    IS_TASK_DEP = True

    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            return 10
        from airflow.models.skipmixin import XCOM_SKIPMIXIN_FOLLOWED, XCOM_SKIPMIXIN_KEY, XCOM_SKIPMIXIN_SKIPPED, SkipMixin
        from airflow.utils.state import TaskInstanceState
        upstream = ti.task.get_direct_relatives(upstream=True)
        finished_tis = dep_context.ensure_finished_tis(ti.get_dagrun(session), session)
        finished_task_ids = {t.task_id for t in finished_tis}
        for parent in upstream:
            if isinstance(parent, SkipMixin):
                if parent.task_id not in finished_task_ids:
                    continue
                prev_result = ti.xcom_pull(task_ids=parent.task_id, key=XCOM_SKIPMIXIN_KEY, session=session)
                if prev_result is None:
                    continue
                should_skip = False
                if XCOM_SKIPMIXIN_FOLLOWED in prev_result and ti.task_id not in prev_result[XCOM_SKIPMIXIN_FOLLOWED]:
                    should_skip = True
                elif XCOM_SKIPMIXIN_SKIPPED in prev_result and ti.task_id in prev_result[XCOM_SKIPMIXIN_SKIPPED]:
                    should_skip = True
                if should_skip:
                    if dep_context.wait_for_past_depends_before_skipping:
                        past_depends_met = ti.xcom_pull(task_ids=ti.task_id, key=PAST_DEPENDS_MET, session=session, default=False)
                        if not past_depends_met:
                            yield self._failing_status(reason='Task should be skipped but the past depends are not met')
                            return
                    ti.set_state(TaskInstanceState.SKIPPED, session)
                    yield self._failing_status(reason=f'Skipping because of previous XCom result from parent task {parent.task_id}')
                    return