from __future__ import annotations
import collections.abc
import functools
from collections import Counter
from typing import TYPE_CHECKING, Iterator, KeysView, NamedTuple
from sqlalchemy import and_, func, or_, select
from airflow.models.taskinstance import PAST_DEPENDS_MET
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.state import TaskInstanceState
from airflow.utils.trigger_rule import TriggerRule as TR
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from sqlalchemy.sql.expression import ColumnOperators
    from airflow import DAG
    from airflow.models.taskinstance import TaskInstance
    from airflow.ti_deps.dep_context import DepContext
    from airflow.ti_deps.deps.base_ti_dep import TIDepStatus

class _UpstreamTIStates(NamedTuple):
    """States of the upstream tis for a specific ti.

    This is used to determine whether the specific ti can run in this iteration.
    """
    success: int
    skipped: int
    failed: int
    upstream_failed: int
    removed: int
    done: int
    success_setup: int
    skipped_setup: int

    @classmethod
    def calculate(cls, finished_upstreams: Iterator[TaskInstance]) -> _UpstreamTIStates:
        if False:
            for i in range(10):
                print('nop')
        'Calculate states for a task instance.\n\n        ``counter`` is inclusive of ``setup_counter`` -- e.g. if there are 2 skipped upstreams, one\n        of which is a setup, then counter will show 2 skipped and setup counter will show 1.\n\n        :param ti: the ti that we want to calculate deps for\n        :param finished_tis: all the finished tasks of the dag_run\n        '
        counter: dict[str, int] = Counter()
        setup_counter: dict[str, int] = Counter()
        for ti in finished_upstreams:
            curr_state = {ti.state: 1}
            counter.update(curr_state)
            if ti.task.is_setup:
                setup_counter.update(curr_state)
        return _UpstreamTIStates(success=counter.get(TaskInstanceState.SUCCESS, 0), skipped=counter.get(TaskInstanceState.SKIPPED, 0), failed=counter.get(TaskInstanceState.FAILED, 0), upstream_failed=counter.get(TaskInstanceState.UPSTREAM_FAILED, 0), removed=counter.get(TaskInstanceState.REMOVED, 0), done=sum(counter.values()), success_setup=setup_counter.get(TaskInstanceState.SUCCESS, 0), skipped_setup=setup_counter.get(TaskInstanceState.SKIPPED, 0))

class TriggerRuleDep(BaseTIDep):
    """Determines if a task's upstream tasks are in a state that allows a given task instance to run."""
    NAME = 'Trigger Rule'
    IGNORABLE = True
    IS_TASK_DEP = True

    def _get_dep_statuses(self, ti: TaskInstance, session: Session, dep_context: DepContext) -> Iterator[TIDepStatus]:
        if False:
            while True:
                i = 10
        if not ti.task.upstream_task_ids:
            yield self._passing_status(reason='The task instance did not have any upstream tasks.')
            return
        if ti.task.trigger_rule == TR.ALWAYS:
            yield self._passing_status(reason='The task had a always trigger rule set.')
            return
        yield from self._evaluate_trigger_rule(ti=ti, dep_context=dep_context, session=session)

    def _evaluate_trigger_rule(self, *, ti: TaskInstance, dep_context: DepContext, session: Session) -> Iterator[TIDepStatus]:
        if False:
            return 10
        "Evaluate whether ``ti``'s trigger rule was met.\n\n        :param ti: Task instance to evaluate the trigger rule of.\n        :param dep_context: The current dependency context.\n        :param session: Database session.\n        "
        from airflow.models.abstractoperator import NotMapped
        from airflow.models.expandinput import NotFullyPopulated
        from airflow.models.operator import needs_expansion
        from airflow.models.taskinstance import TaskInstance

        @functools.lru_cache
        def _get_expanded_ti_count() -> int:
            if False:
                return 10
            'Get how many tis the current task is supposed to be expanded into.\n\n            This extra closure allows us to query the database only when needed,\n            and at most once.\n            '
            return ti.task.get_mapped_ti_count(ti.run_id, session=session)

        @functools.lru_cache
        def _get_relevant_upstream_map_indexes(upstream_id: str) -> int | range | None:
            if False:
                for i in range(10):
                    print('nop')
            "Get the given task's map indexes relevant to the current ti.\n\n            This extra closure allows us to query the database only when needed,\n            and at most once for each task (instead of once for each expanded\n            task instance of the same task).\n            "
            if TYPE_CHECKING:
                assert isinstance(ti.task.dag, DAG)
            try:
                expanded_ti_count = _get_expanded_ti_count()
            except (NotFullyPopulated, NotMapped):
                return None
            return ti.get_relevant_upstream_map_indexes(upstream=ti.task.dag.task_dict[upstream_id], ti_count=expanded_ti_count, session=session)

        def _is_relevant_upstream(upstream: TaskInstance, relevant_ids: set[str] | KeysView[str]) -> bool:
            if False:
                while True:
                    i = 10
            '\n            Whether a task instance is a "relevant upstream" of the current task.\n\n            This will return false if upstream.task_id is not in relevant_ids,\n            or if both of the following are true:\n                1. upstream.task_id in relevant_ids is True\n                2. ti is in a mapped task group and upstream has a map index\n                  that ti does not depend on.\n            '
            if upstream.task_id not in relevant_ids:
                return False
            if ti.task.get_closest_mapped_task_group() is None:
                return True
            if upstream.map_index < 0:
                return True
            relevant = _get_relevant_upstream_map_indexes(upstream_id=upstream.task_id)
            if relevant is None:
                return True
            if relevant == upstream.map_index:
                return True
            if isinstance(relevant, collections.abc.Container) and upstream.map_index in relevant:
                return True
            return False

        def _iter_upstream_conditions(relevant_tasks: dict) -> Iterator[ColumnOperators]:
            if False:
                for i in range(10):
                    print('nop')
            from airflow.models.taskinstance import TaskInstance
            if ti.task.get_closest_mapped_task_group() is None:
                yield TaskInstance.task_id.in_(relevant_tasks.keys())
                return
            for upstream_id in relevant_tasks:
                map_indexes = _get_relevant_upstream_map_indexes(upstream_id)
                if map_indexes is None:
                    yield (TaskInstance.task_id == upstream_id)
                    continue
                yield and_(TaskInstance.task_id == upstream_id, TaskInstance.map_index < 0)
                if isinstance(map_indexes, range) and map_indexes.step == 1:
                    yield and_(TaskInstance.task_id == upstream_id, TaskInstance.map_index >= map_indexes.start, TaskInstance.map_index < map_indexes.stop)
                elif isinstance(map_indexes, collections.abc.Container):
                    yield and_(TaskInstance.task_id == upstream_id, TaskInstance.map_index.in_(map_indexes))
                else:
                    yield and_(TaskInstance.task_id == upstream_id, TaskInstance.map_index == map_indexes)

        def _evaluate_setup_constraint(*, relevant_setups) -> Iterator[tuple[TIDepStatus, bool]]:
            if False:
                return 10
            "Evaluate whether ``ti``'s trigger rule was met.\n\n            :param ti: Task instance to evaluate the trigger rule of.\n            :param dep_context: The current dependency context.\n            :param session: Database session.\n            "
            task = ti.task
            indirect_setups = {k: v for (k, v) in relevant_setups.items() if k not in task.upstream_task_ids}
            finished_upstream_tis = (x for x in dep_context.ensure_finished_tis(ti.get_dagrun(session), session) if _is_relevant_upstream(upstream=x, relevant_ids=indirect_setups.keys()))
            upstream_states = _UpstreamTIStates.calculate(finished_upstream_tis)
            success = upstream_states.success
            skipped = upstream_states.skipped
            failed = upstream_states.failed
            upstream_failed = upstream_states.upstream_failed
            removed = upstream_states.removed
            if not any((needs_expansion(t) for t in indirect_setups.values())):
                upstream = len(indirect_setups)
            else:
                task_id_counts = session.execute(select(TaskInstance.task_id, func.count(TaskInstance.task_id)).where(TaskInstance.dag_id == ti.dag_id, TaskInstance.run_id == ti.run_id).where(or_(*_iter_upstream_conditions(relevant_tasks=indirect_setups))).group_by(TaskInstance.task_id)).all()
                upstream = sum((count for (_, count) in task_id_counts))
            new_state = None
            changed = False
            if dep_context.flag_upstream_failed:
                if upstream_failed or failed:
                    new_state = TaskInstanceState.UPSTREAM_FAILED
                elif skipped:
                    new_state = TaskInstanceState.SKIPPED
                elif removed and success and (ti.map_index > -1):
                    if ti.map_index >= success:
                        new_state = TaskInstanceState.REMOVED
            if new_state is not None:
                if new_state == TaskInstanceState.SKIPPED and dep_context.wait_for_past_depends_before_skipping:
                    past_depends_met = ti.xcom_pull(task_ids=ti.task_id, key=PAST_DEPENDS_MET, session=session, default=False)
                    if not past_depends_met:
                        yield (self._failing_status(reason='Task should be skipped but the past depends are not met'), changed)
                        return
                changed = ti.set_state(new_state, session)
            if changed:
                dep_context.have_changed_ti_states = True
            non_successes = upstream - success
            if ti.map_index > -1:
                non_successes -= removed
            if non_successes > 0:
                yield (self._failing_status(reason=f'All setup tasks must complete successfully. Relevant setups: {relevant_setups}: upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}'), changed)

        def _evaluate_direct_relatives() -> Iterator[TIDepStatus]:
            if False:
                print('Hello World!')
            "Evaluate whether ``ti``'s trigger rule was met.\n\n            :param ti: Task instance to evaluate the trigger rule of.\n            :param dep_context: The current dependency context.\n            :param session: Database session.\n            "
            task = ti.task
            upstream_tasks = {t.task_id: t for t in task.upstream_list}
            trigger_rule = task.trigger_rule
            finished_upstream_tis = (finished_ti for finished_ti in dep_context.ensure_finished_tis(ti.get_dagrun(session), session) if _is_relevant_upstream(upstream=finished_ti, relevant_ids=ti.task.upstream_task_ids))
            upstream_states = _UpstreamTIStates.calculate(finished_upstream_tis)
            success = upstream_states.success
            skipped = upstream_states.skipped
            failed = upstream_states.failed
            upstream_failed = upstream_states.upstream_failed
            removed = upstream_states.removed
            done = upstream_states.done
            success_setup = upstream_states.success_setup
            skipped_setup = upstream_states.skipped_setup
            if not any((needs_expansion(t) for t in upstream_tasks.values())):
                upstream = len(upstream_tasks)
                upstream_setup = sum((1 for x in upstream_tasks.values() if x.is_setup))
            else:
                task_id_counts = session.execute(select(TaskInstance.task_id, func.count(TaskInstance.task_id)).where(TaskInstance.dag_id == ti.dag_id, TaskInstance.run_id == ti.run_id).where(or_(*_iter_upstream_conditions(relevant_tasks=upstream_tasks))).group_by(TaskInstance.task_id)).all()
                upstream = sum((count for (_, count) in task_id_counts))
                upstream_setup = sum((c for (t, c) in task_id_counts if upstream_tasks[t].is_setup))
            upstream_done = done >= upstream
            changed = False
            new_state = None
            if dep_context.flag_upstream_failed:
                if trigger_rule == TR.ALL_SUCCESS:
                    if upstream_failed or failed:
                        new_state = TaskInstanceState.UPSTREAM_FAILED
                    elif skipped:
                        new_state = TaskInstanceState.SKIPPED
                    elif removed and success and (ti.map_index > -1):
                        if ti.map_index >= success:
                            new_state = TaskInstanceState.REMOVED
                elif trigger_rule == TR.ALL_FAILED:
                    if success or skipped:
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.ONE_SUCCESS:
                    if upstream_done and done == skipped:
                        new_state = TaskInstanceState.SKIPPED
                    elif upstream_done and success <= 0:
                        new_state = TaskInstanceState.UPSTREAM_FAILED
                elif trigger_rule == TR.ONE_FAILED:
                    if upstream_done and (not (failed or upstream_failed)):
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.ONE_DONE:
                    if upstream_done and (not (failed or success)):
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.NONE_FAILED:
                    if upstream_failed or failed:
                        new_state = TaskInstanceState.UPSTREAM_FAILED
                elif trigger_rule == TR.NONE_FAILED_MIN_ONE_SUCCESS:
                    if upstream_failed or failed:
                        new_state = TaskInstanceState.UPSTREAM_FAILED
                    elif skipped == upstream:
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.NONE_SKIPPED:
                    if skipped:
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.ALL_SKIPPED:
                    if success or failed or upstream_failed:
                        new_state = TaskInstanceState.SKIPPED
                elif trigger_rule == TR.ALL_DONE_SETUP_SUCCESS:
                    if upstream_done and upstream_setup and (skipped_setup >= upstream_setup):
                        new_state = TaskInstanceState.SKIPPED
                    elif upstream_done and upstream_setup and (success_setup == 0):
                        new_state = TaskInstanceState.UPSTREAM_FAILED
            if new_state is not None:
                if new_state == TaskInstanceState.SKIPPED and dep_context.wait_for_past_depends_before_skipping:
                    past_depends_met = ti.xcom_pull(task_ids=ti.task_id, key=PAST_DEPENDS_MET, session=session, default=False)
                    if not past_depends_met:
                        yield self._failing_status(reason='Task should be skipped but the past depends are not met')
                        return
                changed = ti.set_state(new_state, session)
            if changed:
                dep_context.have_changed_ti_states = True
            if trigger_rule == TR.ONE_SUCCESS:
                if success <= 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires one upstream task success, but none were found. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ONE_FAILED:
                if not failed and (not upstream_failed):
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires one upstream task failure, but none were found. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ONE_DONE:
                if success + failed <= 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}'requires at least one upstream task failure or successbut none were failed or success. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ALL_SUCCESS:
                num_failures = upstream - success
                if ti.map_index > -1:
                    num_failures -= removed
                if num_failures > 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have succeeded, but found {num_failures} non-success(es). upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ALL_FAILED:
                num_success = upstream - failed - upstream_failed
                if ti.map_index > -1:
                    num_success -= removed
                if num_success > 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have failed, but found {num_success} non-failure(s). upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ALL_DONE:
                if not upstream_done:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have completed, but found {len(upstream_tasks) - done} task(s) that were not done. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.NONE_FAILED or trigger_rule == TR.NONE_FAILED_MIN_ONE_SUCCESS:
                num_failures = upstream - success - skipped
                if ti.map_index > -1:
                    num_failures -= removed
                if num_failures > 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have succeeded or been skipped, but found {num_failures} non-success(es). upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.NONE_SKIPPED:
                if not upstream_done or skipped > 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to not have been skipped, but found {skipped} task(s) skipped. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ALL_SKIPPED:
                num_non_skipped = upstream - skipped
                if num_non_skipped > 0:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have been skipped, but found {num_non_skipped} task(s) in non skipped state. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            elif trigger_rule == TR.ALL_DONE_SETUP_SUCCESS:
                if not upstream_done:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires all upstream tasks to have completed, but found {len(upstream_tasks) - done} task(s) that were not done. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
                elif upstream_setup is None:
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' cannot have mapped tasks as upstream. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
                elif upstream_setup and (not success_setup):
                    yield self._failing_status(reason=f"Task's trigger rule '{trigger_rule}' requires at least one upstream setup task be successful, but found {upstream_setup - success_setup} task(s) that were not. upstream_states={upstream_states}, upstream_task_ids={task.upstream_task_ids}")
            else:
                yield self._failing_status(reason=f"No strategy to evaluate trigger rule '{trigger_rule}'.")
        if not ti.task.is_teardown:
            relevant_setups = {t.task_id: t for t in ti.task.get_upstreams_only_setups()}
            if relevant_setups:
                for (status, changed) in _evaluate_setup_constraint(relevant_setups=relevant_setups):
                    yield status
                    if not status.passed and changed:
                        return
        yield from _evaluate_direct_relatives()