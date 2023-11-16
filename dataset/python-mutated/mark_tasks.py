"""Marks tasks APIs."""
from __future__ import annotations
from typing import TYPE_CHECKING, Collection, Iterable, Iterator, NamedTuple
from sqlalchemy import or_, select
from sqlalchemy.orm import lazyload
from airflow.models.dagrun import DagRun
from airflow.models.taskinstance import TaskInstance
from airflow.operators.subdag import SubDagOperator
from airflow.utils import timezone
from airflow.utils.helpers import exactly_one
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.state import DagRunState, State, TaskInstanceState
from airflow.utils.types import DagRunType
if TYPE_CHECKING:
    from datetime import datetime
    from sqlalchemy.orm import Session as SASession
    from airflow.models.dag import DAG
    from airflow.models.operator import Operator

class _DagRunInfo(NamedTuple):
    logical_date: datetime
    data_interval: tuple[datetime, datetime]

def _create_dagruns(dag: DAG, infos: Iterable[_DagRunInfo], state: DagRunState, run_type: DagRunType) -> Iterable[DagRun]:
    if False:
        return 10
    'Infers from data intervals which DAG runs need to be created and does so.\n\n    :param dag: The DAG to create runs for.\n    :param infos: List of logical dates and data intervals to evaluate.\n    :param state: The state to set the dag run to\n    :param run_type: The prefix will be used to construct dag run id: ``{run_id_prefix}__{execution_date}``.\n    :return: Newly created and existing dag runs for the execution dates supplied.\n    '
    dag_runs = {run.logical_date: run for run in DagRun.find(dag_id=dag.dag_id, execution_date=[info.logical_date for info in infos])}
    for info in infos:
        if info.logical_date not in dag_runs:
            dag_runs[info.logical_date] = dag.create_dagrun(execution_date=info.logical_date, data_interval=info.data_interval, start_date=timezone.utcnow(), external_trigger=False, state=state, run_type=run_type)
    return dag_runs.values()

@provide_session
def set_state(*, tasks: Collection[Operator | tuple[Operator, int]], run_id: str | None=None, execution_date: datetime | None=None, upstream: bool=False, downstream: bool=False, future: bool=False, past: bool=False, state: TaskInstanceState=TaskInstanceState.SUCCESS, commit: bool=False, session: SASession=NEW_SESSION) -> list[TaskInstance]:
    if False:
        print('Hello World!')
    '\n    Set the state of a task instance and if needed its relatives.\n\n    Can set state for future tasks (calculated from run_id) and retroactively\n    for past tasks. Will verify integrity of past dag runs in order to create\n    tasks that did not exist. It will not create dag runs that are missing\n    on the schedule (but it will, as for subdag, dag runs if needed).\n\n    :param tasks: the iterable of tasks or (task, map_index) tuples from which to work.\n        ``task.dag`` needs to be set\n    :param run_id: the run_id of the dagrun to start looking from\n    :param execution_date: the execution date from which to start looking (deprecated)\n    :param upstream: Mark all parents (upstream tasks)\n    :param downstream: Mark all siblings (downstream tasks) of task_id, including SubDags\n    :param future: Mark all future tasks on the interval of the dag up until\n        last execution date.\n    :param past: Retroactively mark all tasks starting from start_date of the DAG\n    :param state: State to which the tasks need to be set\n    :param commit: Commit tasks to be altered to the database\n    :param session: database session\n    :return: list of tasks that have been created and updated\n    '
    if not tasks:
        return []
    if not exactly_one(execution_date, run_id):
        raise ValueError('Exactly one of dag_run_id and execution_date must be set')
    if execution_date and (not timezone.is_localized(execution_date)):
        raise ValueError(f'Received non-localized date {execution_date}')
    task_dags = {task[0].dag if isinstance(task, tuple) else task.dag for task in tasks}
    if len(task_dags) > 1:
        raise ValueError(f'Received tasks from multiple DAGs: {task_dags}')
    dag = next(iter(task_dags))
    if dag is None:
        raise ValueError('Received tasks with no DAG')
    if execution_date:
        run_id = dag.get_dagrun(execution_date=execution_date, session=session).run_id
    if not run_id:
        raise ValueError('Received tasks with no run_id')
    dag_run_ids = get_run_ids(dag, run_id, future, past, session=session)
    task_id_map_index_list = list(find_task_relatives(tasks, downstream, upstream))
    task_ids = [task_id if isinstance(task_id, str) else task_id[0] for task_id in task_id_map_index_list]
    confirmed_infos = list(_iter_existing_dag_run_infos(dag, dag_run_ids, session=session))
    confirmed_dates = [info.logical_date for info in confirmed_infos]
    sub_dag_run_ids = list(_iter_subdag_run_ids(dag, session, DagRunState(state), task_ids, commit, confirmed_infos))
    qry_dag = get_all_dag_task_query(dag, session, state, task_id_map_index_list, dag_run_ids)
    if commit:
        tis_altered = session.scalars(qry_dag.with_for_update()).all()
        if sub_dag_run_ids:
            qry_sub_dag = all_subdag_tasks_query(sub_dag_run_ids, session, state, confirmed_dates)
            tis_altered += session.scalars(qry_sub_dag.with_for_update()).all()
        for task_instance in tis_altered:
            if task_instance.state in (TaskInstanceState.DEFERRED, TaskInstanceState.UP_FOR_RESCHEDULE):
                task_instance._try_number += 1
            task_instance.set_state(state, session=session)
        session.flush()
    else:
        tis_altered = session.scalars(qry_dag).all()
        if sub_dag_run_ids:
            qry_sub_dag = all_subdag_tasks_query(sub_dag_run_ids, session, state, confirmed_dates)
            tis_altered += session.scalars(qry_sub_dag).all()
    return tis_altered

def all_subdag_tasks_query(sub_dag_run_ids: list[str], session: SASession, state: TaskInstanceState, confirmed_dates: Iterable[datetime]):
    if False:
        while True:
            i = 10
    'Get *all* tasks of the sub dags.'
    qry_sub_dag = select(TaskInstance).where(TaskInstance.dag_id.in_(sub_dag_run_ids), TaskInstance.execution_date.in_(confirmed_dates)).where(or_(TaskInstance.state.is_(None), TaskInstance.state != state))
    return qry_sub_dag

def get_all_dag_task_query(dag: DAG, session: SASession, state: TaskInstanceState, task_ids: list[str | tuple[str, int]], run_ids: Iterable[str]):
    if False:
        print('Hello World!')
    'Get all tasks of the main dag that will be affected by a state change.'
    qry_dag = select(TaskInstance).where(TaskInstance.dag_id == dag.dag_id, TaskInstance.run_id.in_(run_ids), TaskInstance.ti_selector_condition(task_ids))
    qry_dag = qry_dag.where(or_(TaskInstance.state.is_(None), TaskInstance.state != state)).options(lazyload(TaskInstance.dag_run))
    return qry_dag

def _iter_subdag_run_ids(dag: DAG, session: SASession, state: DagRunState, task_ids: list[str], commit: bool, confirmed_infos: Iterable[_DagRunInfo]) -> Iterator[str]:
    if False:
        return 10
    'Go through subdag operators and create dag runs.\n\n    We only work within the scope of the subdag. A subdag does not propagate to\n    its parent DAG, but parent propagates to subdags.\n    '
    dags = [dag]
    while dags:
        current_dag = dags.pop()
        for task_id in task_ids:
            if not current_dag.has_task(task_id):
                continue
            current_task = current_dag.get_task(task_id)
            if isinstance(current_task, SubDagOperator) or current_task.task_type == 'SubDagOperator':
                if TYPE_CHECKING:
                    assert current_task.subdag
                dag_runs = _create_dagruns(current_task.subdag, infos=confirmed_infos, state=DagRunState.RUNNING, run_type=DagRunType.BACKFILL_JOB)
                verify_dagruns(dag_runs, commit, state, session, current_task)
                dags.append(current_task.subdag)
                yield current_task.subdag.dag_id

def verify_dagruns(dag_runs: Iterable[DagRun], commit: bool, state: DagRunState, session: SASession, current_task: Operator):
    if False:
        print('Hello World!')
    'Verify integrity of dag_runs.\n\n    :param dag_runs: dag runs to verify\n    :param commit: whether dag runs state should be updated\n    :param state: state of the dag_run to set if commit is True\n    :param session: session to use\n    :param current_task: current task\n    '
    for dag_run in dag_runs:
        dag_run.dag = current_task.subdag
        dag_run.verify_integrity()
        if commit:
            dag_run.state = state
            session.merge(dag_run)

def _iter_existing_dag_run_infos(dag: DAG, run_ids: list[str], session: SASession) -> Iterator[_DagRunInfo]:
    if False:
        for i in range(10):
            print('nop')
    for dag_run in DagRun.find(dag_id=dag.dag_id, run_id=run_ids, session=session):
        dag_run.dag = dag
        dag_run.verify_integrity(session=session)
        yield _DagRunInfo(dag_run.logical_date, dag.get_run_data_interval(dag_run))

def find_task_relatives(tasks, downstream, upstream):
    if False:
        return 10
    'Yield task ids and optionally ancestor and descendant ids.'
    for item in tasks:
        if isinstance(item, tuple):
            (task, map_index) = item
            yield (task.task_id, map_index)
        else:
            task = item
            yield task.task_id
        if downstream:
            for relative in task.get_flat_relatives(upstream=False):
                yield relative.task_id
        if upstream:
            for relative in task.get_flat_relatives(upstream=True):
                yield relative.task_id

@provide_session
def get_execution_dates(dag: DAG, execution_date: datetime, future: bool, past: bool, *, session: SASession=NEW_SESSION) -> list[datetime]:
    if False:
        return 10
    'Return DAG execution dates.'
    latest_execution_date = dag.get_latest_execution_date(session=session)
    if latest_execution_date is None:
        raise ValueError(f'Received non-localized date {execution_date}')
    execution_date = timezone.coerce_datetime(execution_date)
    end_date = latest_execution_date if future else execution_date
    if dag.start_date:
        start_date = dag.start_date
    else:
        start_date = execution_date
    start_date = execution_date if not past else start_date
    if not dag.timetable.can_be_scheduled:
        dag_runs = dag.get_dagruns_between(start_date=start_date, end_date=end_date)
        dates = sorted({d.execution_date for d in dag_runs})
    elif not dag.timetable.periodic:
        dates = [start_date]
    else:
        dates = [info.logical_date for info in dag.iter_dagrun_infos_between(start_date, end_date, align=False)]
    return dates

@provide_session
def get_run_ids(dag: DAG, run_id: str, future: bool, past: bool, session: SASession=NEW_SESSION):
    if False:
        while True:
            i = 10
    "Return DAG executions' run_ids."
    last_dagrun = dag.get_last_dagrun(include_externally_triggered=True, session=session)
    current_dagrun = dag.get_dagrun(run_id=run_id, session=session)
    first_dagrun = session.scalar(select(DagRun).filter(DagRun.dag_id == dag.dag_id).order_by(DagRun.execution_date.asc()).limit(1))
    if last_dagrun is None:
        raise ValueError(f'DagRun for {dag.dag_id} not found')
    end_date = last_dagrun.logical_date if future else current_dagrun.logical_date
    start_date = current_dagrun.logical_date if not past else first_dagrun.logical_date
    if not dag.timetable.can_be_scheduled:
        dag_runs = dag.get_dagruns_between(start_date=start_date, end_date=end_date, session=session)
        run_ids = sorted({d.run_id for d in dag_runs})
    elif not dag.timetable.periodic:
        run_ids = [run_id]
    else:
        dates = [info.logical_date for info in dag.iter_dagrun_infos_between(start_date, end_date, align=False)]
        run_ids = [dr.run_id for dr in DagRun.find(dag_id=dag.dag_id, execution_date=dates, session=session)]
    return run_ids

def _set_dag_run_state(dag_id: str, run_id: str, state: DagRunState, session: SASession):
    if False:
        while True:
            i = 10
    '\n    Set dag run state in the DB.\n\n    :param dag_id: dag_id of target dag run\n    :param run_id: run id of target dag run\n    :param state: target state\n    :param session: database session\n    '
    dag_run = session.execute(select(DagRun).where(DagRun.dag_id == dag_id, DagRun.run_id == run_id)).scalar_one()
    dag_run.state = state
    if state == DagRunState.RUNNING:
        dag_run.start_date = timezone.utcnow()
        dag_run.end_date = None
    else:
        dag_run.end_date = timezone.utcnow()
    session.merge(dag_run)

@provide_session
def set_dag_run_state_to_success(*, dag: DAG, execution_date: datetime | None=None, run_id: str | None=None, commit: bool=False, session: SASession=NEW_SESSION) -> list[TaskInstance]:
    if False:
        i = 10
        return i + 15
    "\n    Set the dag run's state to success.\n\n    Set for a specific execution date and its task instances to success.\n\n    :param dag: the DAG of which to alter state\n    :param execution_date: the execution date from which to start looking(deprecated)\n    :param run_id: the run_id to start looking from\n    :param commit: commit DAG and tasks to be altered to the database\n    :param session: database session\n    :return: If commit is true, list of tasks that have been updated,\n             otherwise list of tasks that will be updated\n    :raises: ValueError if dag or execution_date is invalid\n    "
    if not exactly_one(execution_date, run_id):
        return []
    if not dag:
        return []
    if execution_date:
        if not timezone.is_localized(execution_date):
            raise ValueError(f'Received non-localized date {execution_date}')
        dag_run = dag.get_dagrun(execution_date=execution_date)
        if not dag_run:
            raise ValueError(f'DagRun with execution_date: {execution_date} not found')
        run_id = dag_run.run_id
    if not run_id:
        raise ValueError(f'Invalid dag_run_id: {run_id}')
    if commit:
        _set_dag_run_state(dag.dag_id, run_id, DagRunState.SUCCESS, session)
    for task in dag.tasks:
        task.dag = dag
    return set_state(tasks=dag.tasks, run_id=run_id, state=TaskInstanceState.SUCCESS, commit=commit, session=session)

@provide_session
def set_dag_run_state_to_failed(*, dag: DAG, execution_date: datetime | None=None, run_id: str | None=None, commit: bool=False, session: SASession=NEW_SESSION) -> list[TaskInstance]:
    if False:
        while True:
            i = 10
    "\n    Set the dag run's state to failed.\n\n    Set for a specific execution date and its task instances to failed.\n\n    :param dag: the DAG of which to alter state\n    :param execution_date: the execution date from which to start looking(deprecated)\n    :param run_id: the DAG run_id to start looking from\n    :param commit: commit DAG and tasks to be altered to the database\n    :param session: database session\n    :return: If commit is true, list of tasks that have been updated,\n             otherwise list of tasks that will be updated\n    :raises: AssertionError if dag or execution_date is invalid\n    "
    if not exactly_one(execution_date, run_id):
        return []
    if not dag:
        return []
    if execution_date:
        if not timezone.is_localized(execution_date):
            raise ValueError(f'Received non-localized date {execution_date}')
        dag_run = dag.get_dagrun(execution_date=execution_date)
        if not dag_run:
            raise ValueError(f'DagRun with execution_date: {execution_date} not found')
        run_id = dag_run.run_id
    if not run_id:
        raise ValueError(f'Invalid dag_run_id: {run_id}')
    if commit:
        _set_dag_run_state(dag.dag_id, run_id, DagRunState.FAILED, session)
    running_states = (TaskInstanceState.RUNNING, TaskInstanceState.DEFERRED, TaskInstanceState.UP_FOR_RESCHEDULE)
    task_ids = [task.task_id for task in dag.tasks]
    tis = session.scalars(select(TaskInstance).where(TaskInstance.dag_id == dag.dag_id, TaskInstance.run_id == run_id, TaskInstance.task_id.in_(task_ids), TaskInstance.state.in_(running_states)))
    task_ids_of_running_tis = [task_instance.task_id for task_instance in tis]
    tasks = []
    for task in dag.tasks:
        if task.task_id in task_ids_of_running_tis:
            task.dag = dag
            tasks.append(task)
    tis = session.scalars(select(TaskInstance).filter(TaskInstance.dag_id == dag.dag_id, TaskInstance.run_id == run_id, TaskInstance.state.not_in(State.finished), TaskInstance.state.not_in(running_states))).all()
    if commit:
        for ti in tis:
            ti.set_state(TaskInstanceState.SKIPPED)
    return tis + set_state(tasks=tasks, run_id=run_id, state=TaskInstanceState.FAILED, commit=commit, session=session)

def __set_dag_run_state_to_running_or_queued(*, new_state: DagRunState, dag: DAG, execution_date: datetime | None=None, run_id: str | None=None, commit: bool=False, session: SASession) -> list[TaskInstance]:
    if False:
        i = 10
        return i + 15
    '\n    Set the dag run for a specific execution date to running.\n\n    :param dag: the DAG of which to alter state\n    :param execution_date: the execution date from which to start looking\n    :param run_id: the id of the DagRun\n    :param commit: commit DAG and tasks to be altered to the database\n    :param session: database session\n    :return: If commit is true, list of tasks that have been updated,\n             otherwise list of tasks that will be updated\n    '
    res: list[TaskInstance] = []
    if not exactly_one(execution_date, run_id):
        return res
    if not dag:
        return res
    if execution_date:
        if not timezone.is_localized(execution_date):
            raise ValueError(f'Received non-localized date {execution_date}')
        dag_run = dag.get_dagrun(execution_date=execution_date)
        if not dag_run:
            raise ValueError(f'DagRun with execution_date: {execution_date} not found')
        run_id = dag_run.run_id
    if not run_id:
        raise ValueError(f'DagRun with run_id: {run_id} not found')
    if commit:
        _set_dag_run_state(dag.dag_id, run_id, new_state, session)
    return res

@provide_session
def set_dag_run_state_to_running(*, dag: DAG, execution_date: datetime | None=None, run_id: str | None=None, commit: bool=False, session: SASession=NEW_SESSION) -> list[TaskInstance]:
    if False:
        print('Hello World!')
    "\n    Set the dag run's state to running.\n\n    Set for a specific execution date and its task instances to running.\n    "
    return __set_dag_run_state_to_running_or_queued(new_state=DagRunState.RUNNING, dag=dag, execution_date=execution_date, run_id=run_id, commit=commit, session=session)

@provide_session
def set_dag_run_state_to_queued(*, dag: DAG, execution_date: datetime | None=None, run_id: str | None=None, commit: bool=False, session: SASession=NEW_SESSION) -> list[TaskInstance]:
    if False:
        i = 10
        return i + 15
    "\n    Set the dag run's state to queued.\n\n    Set for a specific execution date and its task instances to queued.\n    "
    return __set_dag_run_state_to_running_or_queued(new_state=DagRunState.QUEUED, dag=dag, execution_date=execution_date, run_id=run_id, commit=commit, session=session)