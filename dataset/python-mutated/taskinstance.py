from __future__ import annotations
import collections.abc
import contextlib
import hashlib
import itertools
import logging
import math
import operator
import os
import signal
import warnings
from collections import defaultdict
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Collection, Generator, Iterable, Tuple
from urllib.parse import quote
import dill
import jinja2
import lazy_object_proxy
import pendulum
from jinja2 import TemplateAssertionError, UndefinedError
from sqlalchemy import Column, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, PrimaryKeyConstraint, String, Text, and_, delete, false, func, inspect, or_, text, update
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import reconstructor, relationship
from sqlalchemy.orm.attributes import NO_VALUE, set_committed_value
from sqlalchemy.sql.expression import case
from airflow import settings
from airflow.api_internal.internal_api_call import internal_api_call
from airflow.compat.functools import cache
from airflow.configuration import conf
from airflow.datasets import Dataset
from airflow.datasets.manager import dataset_manager
from airflow.exceptions import AirflowException, AirflowFailException, AirflowRescheduleException, AirflowSensorTimeout, AirflowSkipException, AirflowTaskTimeout, DagRunNotFound, RemovedInAirflow3Warning, TaskDeferred, UnmappableXComLengthPushed, UnmappableXComTypePushed, XComForMappingNotPushed
from airflow.listeners.listener import get_listener_manager
from airflow.models.base import Base, StringID
from airflow.models.dagbag import DagBag
from airflow.models.log import Log
from airflow.models.mappedoperator import MappedOperator
from airflow.models.param import process_params
from airflow.models.taskfail import TaskFail
from airflow.models.taskinstancekey import TaskInstanceKey
from airflow.models.taskmap import TaskMap
from airflow.models.taskreschedule import TaskReschedule
from airflow.models.xcom import LazyXComAccess, XCom
from airflow.plugins_manager import integrate_macros_plugins
from airflow.sentry import Sentry
from airflow.stats import Stats
from airflow.templates import SandboxedEnvironment
from airflow.ti_deps.dep_context import DepContext
from airflow.ti_deps.dependencies_deps import REQUEUEABLE_DEPS, RUNNING_DEPS
from airflow.utils import timezone
from airflow.utils.context import ConnectionAccessor, Context, VariableAccessor, context_merge
from airflow.utils.email import send_email
from airflow.utils.helpers import prune_dict, render_template_to_string
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.module_loading import qualname
from airflow.utils.net import get_hostname
from airflow.utils.operator_helpers import context_to_airflow_vars
from airflow.utils.platform import getuser
from airflow.utils.retries import run_with_db_retries
from airflow.utils.session import NEW_SESSION, create_session, provide_session
from airflow.utils.sqlalchemy import ExecutorConfigType, ExtendedJSON, UtcDateTime, tuple_in_condition, with_row_locks
from airflow.utils.state import DagRunState, JobState, State, TaskInstanceState
from airflow.utils.task_group import MappedTaskGroup
from airflow.utils.task_instance_session import set_current_task_instance_session
from airflow.utils.timeout import timeout
from airflow.utils.xcom import XCOM_RETURN_KEY
TR = TaskReschedule
_CURRENT_CONTEXT: list[Context] = []
log = logging.getLogger(__name__)
if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import PurePath
    from types import TracebackType
    from sqlalchemy.orm.session import Session
    from sqlalchemy.sql.elements import BooleanClauseList
    from sqlalchemy.sql.expression import ColumnOperators
    from airflow.models.abstractoperator import TaskStateChangeCallback
    from airflow.models.baseoperator import BaseOperator
    from airflow.models.dag import DAG, DagModel
    from airflow.models.dagrun import DagRun
    from airflow.models.dataset import DatasetEvent
    from airflow.models.operator import Operator
    from airflow.serialization.pydantic.dag import DagModelPydantic
    from airflow.serialization.pydantic.taskinstance import TaskInstancePydantic
    from airflow.timetables.base import DataInterval
    from airflow.typing_compat import Literal, TypeGuard
    from airflow.utils.task_group import TaskGroup
    hybrid_property = property
else:
    from sqlalchemy.ext.hybrid import hybrid_property
PAST_DEPENDS_MET = 'past_depends_met'

class TaskReturnCode(Enum):
    """
    Enum to signal manner of exit for task run command.

    :meta private:
    """
    DEFERRED = 100
    'When task exits with deferral to trigger.'

@contextlib.contextmanager
def set_current_context(context: Context) -> Generator[Context, None, None]:
    if False:
        i = 10
        return i + 15
    '\n    Set the current execution context to the provided context object.\n\n    This method should be called once per Task execution, before calling operator.execute.\n    '
    _CURRENT_CONTEXT.append(context)
    try:
        yield context
    finally:
        expected_state = _CURRENT_CONTEXT.pop()
        if expected_state != context:
            log.warning('Current context is not equal to the state at context stack. Expected=%s, got=%s', context, expected_state)

def _stop_remaining_tasks(*, task_instance: TaskInstance | TaskInstancePydantic, session: Session):
    if False:
        i = 10
        return i + 15
    '\n    Stop non-teardown tasks in dag.\n\n    :meta private:\n    '
    if not task_instance.dag_run:
        raise ValueError('``task_instance`` must have ``dag_run`` set')
    tis = task_instance.dag_run.get_task_instances(session=session)
    if TYPE_CHECKING:
        assert isinstance(task_instance.task.dag, DAG)
    for ti in tis:
        if ti.task_id == task_instance.task_id or ti.state in (TaskInstanceState.SUCCESS, TaskInstanceState.FAILED):
            continue
        task = task_instance.task.dag.task_dict[ti.task_id]
        if not task.is_teardown:
            if ti.state == TaskInstanceState.RUNNING:
                log.info("Forcing task %s to fail due to dag's `fail_stop` setting", ti.task_id)
                ti.error(session)
            else:
                log.info("Setting task %s to SKIPPED due to dag's `fail_stop` setting.", ti.task_id)
                ti.set_state(state=TaskInstanceState.SKIPPED, session=session)
        else:
            log.info("Not skipping teardown task '%s'", ti.task_id)

def clear_task_instances(tis: list[TaskInstance], session: Session, activate_dag_runs: None=None, dag: DAG | None=None, dag_run_state: DagRunState | Literal[False]=DagRunState.QUEUED) -> None:
    if False:
        print('Hello World!')
    "\n    Clear a set of task instances, but make sure the running ones get killed.\n\n    Also sets Dagrun's `state` to QUEUED and `start_date` to the time of execution.\n    But only for finished DRs (SUCCESS and FAILED).\n    Doesn't clear DR's `state` and `start_date`for running\n    DRs (QUEUED and RUNNING) because clearing the state for already\n    running DR is redundant and clearing `start_date` affects DR's duration.\n\n    :param tis: a list of task instances\n    :param session: current session\n    :param dag_run_state: state to set finished DagRuns to.\n        If set to False, DagRuns state will not be changed.\n    :param dag: DAG object\n    :param activate_dag_runs: Deprecated parameter, do not pass\n    "
    job_ids = []
    task_id_by_key: dict[str, dict[str, dict[int, dict[int, set[str]]]]] = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(set))))
    dag_bag = DagBag(read_dags_from_db=True)
    for ti in tis:
        if ti.state == TaskInstanceState.RUNNING:
            if ti.job_id:
                ti.state = TaskInstanceState.RESTARTING
                job_ids.append(ti.job_id)
        else:
            ti_dag = dag if dag and dag.dag_id == ti.dag_id else dag_bag.get_dag(ti.dag_id, session=session)
            task_id = ti.task_id
            if ti_dag and ti_dag.has_task(task_id):
                task = ti_dag.get_task(task_id)
                ti.refresh_from_task(task)
                task_retries = task.retries
                ti.max_tries = ti.try_number + task_retries - 1
            else:
                ti.max_tries = max(ti.max_tries, ti.prev_attempted_tries)
            ti.state = None
            ti.external_executor_id = None
            ti.clear_next_method_args()
            session.merge(ti)
        task_id_by_key[ti.dag_id][ti.run_id][ti.map_index][ti.try_number].add(ti.task_id)
    if task_id_by_key:
        conditions = or_((and_(TR.dag_id == dag_id, or_((and_(TR.run_id == run_id, or_((and_(TR.map_index == map_index, or_((and_(TR.try_number == try_number, TR.task_id.in_(task_ids)) for (try_number, task_ids) in task_tries.items()))) for (map_index, task_tries) in map_indexes.items()))) for (run_id, map_indexes) in run_ids.items()))) for (dag_id, run_ids) in task_id_by_key.items()))
        delete_qry = TR.__table__.delete().where(conditions)
        session.execute(delete_qry)
    if job_ids:
        from airflow.jobs.job import Job
        session.execute(update(Job).where(Job.id.in_(job_ids)).values(state=JobState.RESTARTING))
    if activate_dag_runs is not None:
        warnings.warn('`activate_dag_runs` parameter to clear_task_instances function is deprecated. Please use `dag_run_state`', RemovedInAirflow3Warning, stacklevel=2)
        if not activate_dag_runs:
            dag_run_state = False
    if dag_run_state is not False and tis:
        from airflow.models.dagrun import DagRun
        run_ids_by_dag_id = defaultdict(set)
        for instance in tis:
            run_ids_by_dag_id[instance.dag_id].add(instance.run_id)
        drs = session.query(DagRun).filter(or_((and_(DagRun.dag_id == dag_id, DagRun.run_id.in_(run_ids)) for (dag_id, run_ids) in run_ids_by_dag_id.items()))).all()
        dag_run_state = DagRunState(dag_run_state)
        for dr in drs:
            if dr.state in State.finished_dr_states:
                dr.state = dag_run_state
                dr.start_date = timezone.utcnow()
                if dag_run_state == DagRunState.QUEUED:
                    dr.last_scheduling_decision = None
                    dr.start_date = None
                    dr.clear_number += 1
    session.flush()

def _is_mappable_value(value: Any) -> TypeGuard[Collection]:
    if False:
        for i in range(10):
            print('nop')
    "Whether a value can be used for task mapping.\n\n    We only allow collections with guaranteed ordering, but exclude character\n    sequences since that's usually not what users would expect to be mappable.\n    "
    if not isinstance(value, (collections.abc.Sequence, dict)):
        return False
    if isinstance(value, (bytearray, bytes, str)):
        return False
    return True

def _creator_note(val):
    if False:
        for i in range(10):
            print('nop')
    'Creator the ``note`` association proxy.'
    if isinstance(val, str):
        return TaskInstanceNote(content=val)
    elif isinstance(val, dict):
        return TaskInstanceNote(**val)
    else:
        return TaskInstanceNote(*val)

def _execute_task(task_instance, context, task_orig):
    if False:
        for i in range(10):
            print('nop')
    '\n    Execute Task (optionally with a Timeout) and push Xcom results.\n\n    :param task_instance: the task instance\n    :param context: Jinja2 context\n    :param task_orig: origin task\n\n    :meta private:\n    '
    task_to_execute = task_instance.task
    if isinstance(task_to_execute, MappedOperator):
        raise AirflowException('MappedOperator cannot be executed.')
    execute_callable_kwargs = {}
    if task_instance.next_method:
        if task_instance.next_method:
            execute_callable = task_to_execute.resume_execution
            execute_callable_kwargs['next_method'] = task_instance.next_method
            execute_callable_kwargs['next_kwargs'] = task_instance.next_kwargs
    else:
        execute_callable = task_to_execute.execute
    if task_to_execute.execution_timeout:
        if task_instance.next_method:
            timeout_seconds = (task_to_execute.execution_timeout - (timezone.utcnow() - task_instance.start_date)).total_seconds()
        else:
            timeout_seconds = task_to_execute.execution_timeout.total_seconds()
        try:
            if timeout_seconds <= 0:
                raise AirflowTaskTimeout()
            with timeout(timeout_seconds):
                result = execute_callable(context=context, **execute_callable_kwargs)
        except AirflowTaskTimeout:
            task_to_execute.on_kill()
            raise
    else:
        result = execute_callable(context=context, **execute_callable_kwargs)
    with create_session() as session:
        if task_to_execute.do_xcom_push:
            xcom_value = result
        else:
            xcom_value = None
        if xcom_value is not None:
            task_instance.xcom_push(key=XCOM_RETURN_KEY, value=xcom_value, session=session)
        _record_task_map_for_downstreams(task_instance=task_instance, task=task_orig, value=xcom_value, session=session)
    return result

def _refresh_from_db(*, task_instance: TaskInstance | TaskInstancePydantic, session: Session, lock_for_update: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Refreshes the task instance from the database based on the primary key.\n\n    :param task_instance: the task instance\n    :param session: SQLAlchemy ORM Session\n    :param lock_for_update: if True, indicates that the database should\n        lock the TaskInstance (issuing a FOR UPDATE clause) until the\n        session is committed.\n\n    :meta private:\n    '
    if task_instance in session:
        session.refresh(task_instance, TaskInstance.__mapper__.column_attrs.keys())
    ti = TaskInstance.get_task_instance(dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, map_index=task_instance.map_index, select_columns=True, lock_for_update=lock_for_update, session=session)
    if ti:
        task_instance.start_date = ti.start_date
        task_instance.end_date = ti.end_date
        task_instance.duration = ti.duration
        task_instance.state = ti.state
        task_instance.try_number = ti.try_number
        task_instance.max_tries = ti.max_tries
        task_instance.hostname = ti.hostname
        task_instance.unixname = ti.unixname
        task_instance.job_id = ti.job_id
        task_instance.pool = ti.pool
        task_instance.pool_slots = ti.pool_slots or 1
        task_instance.queue = ti.queue
        task_instance.priority_weight = ti.priority_weight
        task_instance.operator = ti.operator
        task_instance.custom_operator_name = ti.custom_operator_name
        task_instance.queued_dttm = ti.queued_dttm
        task_instance.queued_by_job_id = ti.queued_by_job_id
        task_instance.pid = ti.pid
        task_instance.executor_config = ti.executor_config
        task_instance.external_executor_id = ti.external_executor_id
        task_instance.trigger_id = ti.trigger_id
        task_instance.next_method = ti.next_method
        task_instance.next_kwargs = ti.next_kwargs
    else:
        task_instance.state = None

def _set_duration(*, task_instance: TaskInstance | TaskInstancePydantic) -> None:
    if False:
        return 10
    '\n    Set task instance duration.\n\n    :param task_instance: the task instance\n\n    :meta private:\n    '
    if task_instance.end_date and task_instance.start_date:
        task_instance.duration = (task_instance.end_date - task_instance.start_date).total_seconds()
    else:
        task_instance.duration = None
    log.debug('Task Duration set to %s', task_instance.duration)

def _stats_tags(*, task_instance: TaskInstance | TaskInstancePydantic) -> dict[str, str]:
    if False:
        while True:
            i = 10
    '\n    Returns task instance tags.\n\n    :param task_instance: the task instance\n\n    :meta private:\n    '
    return prune_dict({'dag_id': task_instance.dag_id, 'task_id': task_instance.task_id})

def _clear_next_method_args(*, task_instance: TaskInstance | TaskInstancePydantic) -> None:
    if False:
        return 10
    "\n    Ensure we unset next_method and next_kwargs to ensure that any retries don't reuse them.\n\n    :param task_instance: the task instance\n\n    :meta private:\n    "
    log.debug('Clearing next_method and next_kwargs.')
    task_instance.next_method = None
    task_instance.next_kwargs = None

def _get_template_context(*, task_instance, session: Session | None=None, ignore_param_exceptions: bool=True) -> Context:
    if False:
        return 10
    '\n    Return TI Context.\n\n    :param task_instance: the task instance\n    :param session: SQLAlchemy ORM Session\n    :param ignore_param_exceptions: flag to suppress value exceptions while initializing the ParamsDict\n\n    :meta private:\n    '
    if not session:
        session = settings.Session()
    from airflow import macros
    from airflow.models.abstractoperator import NotMapped
    integrate_macros_plugins()
    task = task_instance.task
    if TYPE_CHECKING:
        assert task.dag
    dag: DAG = task.dag
    dag_run = task_instance.get_dagrun(session)
    data_interval = dag.get_run_data_interval(dag_run)
    validated_params = process_params(dag, task, dag_run, suppress_exception=ignore_param_exceptions)
    logical_date = timezone.coerce_datetime(task_instance.execution_date)
    ds = logical_date.strftime('%Y-%m-%d')
    ds_nodash = ds.replace('-', '')
    ts = logical_date.isoformat()
    ts_nodash = logical_date.strftime('%Y%m%dT%H%M%S')
    ts_nodash_with_tz = ts.replace('-', '').replace(':', '')

    @cache
    def _get_previous_dagrun_success() -> DagRun | None:
        if False:
            for i in range(10):
                print('nop')
        return task_instance.get_previous_dagrun(state=DagRunState.SUCCESS, session=session)

    def _get_previous_dagrun_data_interval_success() -> DataInterval | None:
        if False:
            i = 10
            return i + 15
        dagrun = _get_previous_dagrun_success()
        if dagrun is None:
            return None
        return dag.get_run_data_interval(dagrun)

    def get_prev_data_interval_start_success() -> pendulum.DateTime | None:
        if False:
            for i in range(10):
                print('nop')
        data_interval = _get_previous_dagrun_data_interval_success()
        if data_interval is None:
            return None
        return data_interval.start

    def get_prev_data_interval_end_success() -> pendulum.DateTime | None:
        if False:
            print('Hello World!')
        data_interval = _get_previous_dagrun_data_interval_success()
        if data_interval is None:
            return None
        return data_interval.end

    def get_prev_start_date_success() -> pendulum.DateTime | None:
        if False:
            return 10
        dagrun = _get_previous_dagrun_success()
        if dagrun is None:
            return None
        return timezone.coerce_datetime(dagrun.start_date)

    def get_prev_end_date_success() -> pendulum.DateTime | None:
        if False:
            return 10
        dagrun = _get_previous_dagrun_success()
        if dagrun is None:
            return None
        return timezone.coerce_datetime(dagrun.end_date)

    @cache
    def get_yesterday_ds() -> str:
        if False:
            i = 10
            return i + 15
        return (logical_date - timedelta(1)).strftime('%Y-%m-%d')

    def get_yesterday_ds_nodash() -> str:
        if False:
            for i in range(10):
                print('nop')
        return get_yesterday_ds().replace('-', '')

    @cache
    def get_tomorrow_ds() -> str:
        if False:
            for i in range(10):
                print('nop')
        return (logical_date + timedelta(1)).strftime('%Y-%m-%d')

    def get_tomorrow_ds_nodash() -> str:
        if False:
            while True:
                i = 10
        return get_tomorrow_ds().replace('-', '')

    @cache
    def get_next_execution_date() -> pendulum.DateTime | None:
        if False:
            while True:
                i = 10
        if dag_run.external_trigger:
            return logical_date
        if dag is None:
            return None
        next_info = dag.next_dagrun_info(data_interval, restricted=False)
        if next_info is None:
            return None
        return timezone.coerce_datetime(next_info.logical_date)

    def get_next_ds() -> str | None:
        if False:
            return 10
        execution_date = get_next_execution_date()
        if execution_date is None:
            return None
        return execution_date.strftime('%Y-%m-%d')

    def get_next_ds_nodash() -> str | None:
        if False:
            for i in range(10):
                print('nop')
        ds = get_next_ds()
        if ds is None:
            return ds
        return ds.replace('-', '')

    @cache
    def get_prev_execution_date():
        if False:
            for i in range(10):
                print('nop')
        if dag_run.external_trigger:
            return logical_date
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RemovedInAirflow3Warning)
            return dag.previous_schedule(logical_date)

    @cache
    def get_prev_ds() -> str | None:
        if False:
            i = 10
            return i + 15
        execution_date = get_prev_execution_date()
        if execution_date is None:
            return None
        return execution_date.strftime('%Y-%m-%d')

    def get_prev_ds_nodash() -> str | None:
        if False:
            while True:
                i = 10
        prev_ds = get_prev_ds()
        if prev_ds is None:
            return None
        return prev_ds.replace('-', '')

    def get_triggering_events() -> dict[str, list[DatasetEvent]]:
        if False:
            i = 10
            return i + 15
        if TYPE_CHECKING:
            assert session is not None
        nonlocal dag_run
        if dag_run not in session:
            dag_run = session.merge(dag_run, load=False)
        dataset_events = dag_run.consumed_dataset_events
        triggering_events: dict[str, list[DatasetEvent]] = defaultdict(list)
        for event in dataset_events:
            triggering_events[event.dataset.uri].append(event)
        return triggering_events
    try:
        expanded_ti_count: int | None = task.get_mapped_ti_count(task_instance.run_id, session=session)
    except NotMapped:
        expanded_ti_count = None
    context = {'conf': conf, 'dag': dag, 'dag_run': dag_run, 'data_interval_end': timezone.coerce_datetime(data_interval.end), 'data_interval_start': timezone.coerce_datetime(data_interval.start), 'ds': ds, 'ds_nodash': ds_nodash, 'execution_date': logical_date, 'expanded_ti_count': expanded_ti_count, 'inlets': task.inlets, 'logical_date': logical_date, 'macros': macros, 'next_ds': get_next_ds(), 'next_ds_nodash': get_next_ds_nodash(), 'next_execution_date': get_next_execution_date(), 'outlets': task.outlets, 'params': validated_params, 'prev_data_interval_start_success': get_prev_data_interval_start_success(), 'prev_data_interval_end_success': get_prev_data_interval_end_success(), 'prev_ds': get_prev_ds(), 'prev_ds_nodash': get_prev_ds_nodash(), 'prev_execution_date': get_prev_execution_date(), 'prev_execution_date_success': task_instance.get_previous_execution_date(state=DagRunState.SUCCESS, session=session), 'prev_start_date_success': get_prev_start_date_success(), 'prev_end_date_success': get_prev_end_date_success(), 'run_id': task_instance.run_id, 'task': task, 'task_instance': task_instance, 'task_instance_key_str': f'{task.dag_id}__{task.task_id}__{ds_nodash}', 'test_mode': task_instance.test_mode, 'ti': task_instance, 'tomorrow_ds': get_tomorrow_ds(), 'tomorrow_ds_nodash': get_tomorrow_ds_nodash(), 'triggering_dataset_events': lazy_object_proxy.Proxy(get_triggering_events), 'ts': ts, 'ts_nodash': ts_nodash, 'ts_nodash_with_tz': ts_nodash_with_tz, 'var': {'json': VariableAccessor(deserialize_json=True), 'value': VariableAccessor(deserialize_json=False)}, 'conn': ConnectionAccessor(), 'yesterday_ds': get_yesterday_ds(), 'yesterday_ds_nodash': get_yesterday_ds_nodash()}
    return Context(context)

def _is_eligible_to_retry(*, task_instance: TaskInstance | TaskInstancePydantic):
    if False:
        for i in range(10):
            print('nop')
    '\n    Is task instance is eligible for retry.\n\n    :param task_instance: the task instance\n\n    :meta private:\n    '
    if task_instance.state == TaskInstanceState.RESTARTING:
        return True
    if not getattr(task_instance, 'task', None):
        return task_instance.try_number <= task_instance.max_tries
    return task_instance.task.retries and task_instance.try_number <= task_instance.max_tries

def _handle_failure(*, task_instance: TaskInstance | TaskInstancePydantic, error: None | str | Exception | KeyboardInterrupt, session: Session, test_mode: bool | None=None, context: Context | None=None, force_fail: bool=False) -> None:
    if False:
        while True:
            i = 10
    "\n    Handle Failure for a task instance.\n\n    :param task_instance: the task instance\n    :param error: if specified, log the specific exception if thrown\n    :param session: SQLAlchemy ORM Session\n    :param test_mode: doesn't record success or failure in the DB if True\n    :param context: Jinja2 context\n    :param force_fail: if True, task does not retry\n\n    :meta private:\n    "
    if test_mode is None:
        test_mode = task_instance.test_mode
    failure_context = TaskInstance.fetch_handle_failure_context(ti=task_instance, error=error, test_mode=test_mode, context=context, force_fail=force_fail, session=session)
    _log_state(task_instance=task_instance, lead_msg='Immediate failure requested. ' if force_fail else '')
    if failure_context['task'] and failure_context['email_for_state'](failure_context['task']) and failure_context['task'].email:
        try:
            task_instance.email_alert(error, failure_context['task'])
        except Exception:
            log.exception('Failed to send email to: %s', failure_context['task'].email)
    if failure_context['callbacks'] and failure_context['context']:
        _run_finished_callback(callbacks=failure_context['callbacks'], context=failure_context['context'])
    if not test_mode:
        TaskInstance.save_to_db(failure_context['ti'], session)

def _get_try_number(*, task_instance: TaskInstance | TaskInstancePydantic):
    if False:
        print('Hello World!')
    '\n    Return the try number that a task number will be when it is actually run.\n\n    If the TaskInstance is currently running, this will match the column in the\n    database, in all other cases this will be incremented.\n\n    This is designed so that task logs end up in the right file.\n\n    :param task_instance: the task instance\n\n    :meta private:\n    '
    if task_instance.state == TaskInstanceState.RUNNING.RUNNING:
        return task_instance._try_number
    return task_instance._try_number + 1

def _set_try_number(*, task_instance: TaskInstance | TaskInstancePydantic, value: int) -> None:
    if False:
        return 10
    '\n    Set a task try number.\n\n    :param task_instance: the task instance\n    :param value: the try number\n\n    :meta private:\n    '
    task_instance._try_number = value

def _refresh_from_task(*, task_instance: TaskInstance | TaskInstancePydantic, task: Operator, pool_override: str | None=None) -> None:
    if False:
        print('Hello World!')
    "\n    Copy common attributes from the given task.\n\n    :param task_instance: the task instance\n    :param task: The task object to copy from\n    :param pool_override: Use the pool_override instead of task's pool\n\n    :meta private:\n    "
    task_instance.task = task
    task_instance.queue = task.queue
    task_instance.pool = pool_override or task.pool
    task_instance.pool_slots = task.pool_slots
    task_instance.priority_weight = task.priority_weight_total
    task_instance.run_as_user = task.run_as_user
    task_instance.executor_config = task.executor_config
    task_instance.operator = task.task_type
    task_instance.custom_operator_name = getattr(task, 'custom_operator_name', None)

def _record_task_map_for_downstreams(*, task_instance: TaskInstance | TaskInstancePydantic, task: Operator, value: Any, session: Session) -> None:
    if False:
        return 10
    '\n    Record the task map for downstream tasks.\n\n    :param task_instance: the task instance\n    :param task: The task object\n    :param value: The value\n    :param session: SQLAlchemy ORM Session\n\n    :meta private:\n    '
    if next(task.iter_mapped_dependants(), None) is None:
        return
    if isinstance(task, MappedOperator):
        return
    if value is None:
        raise XComForMappingNotPushed()
    if not _is_mappable_value(value):
        raise UnmappableXComTypePushed(value)
    task_map = TaskMap.from_task_instance_xcom(task_instance, value)
    max_map_length = conf.getint('core', 'max_map_length', fallback=1024)
    if task_map.length > max_map_length:
        raise UnmappableXComLengthPushed(value, max_map_length)
    session.merge(task_map)

def _get_previous_dagrun(*, task_instance: TaskInstance | TaskInstancePydantic, state: DagRunState | None=None, session: Session | None=None) -> DagRun | None:
    if False:
        return 10
    "\n    The DagRun that ran before this task instance's DagRun.\n\n    :param task_instance: the task instance\n    :param state: If passed, it only take into account instances of a specific state.\n    :param session: SQLAlchemy ORM Session.\n\n    :meta private:\n    "
    dag = task_instance.task.dag
    if dag is None:
        return None
    dr = task_instance.get_dagrun(session=session)
    dr.dag = dag
    from airflow.models.dagrun import DagRun
    ignore_schedule = state is not None or not dag.timetable.can_be_scheduled
    if dag.catchup is True and (not ignore_schedule):
        last_dagrun = DagRun.get_previous_scheduled_dagrun(dr.id, session=session)
    else:
        last_dagrun = DagRun.get_previous_dagrun(dag_run=dr, session=session, state=state)
    if last_dagrun:
        return last_dagrun
    return None

def _get_previous_execution_date(*, task_instance: TaskInstance | TaskInstancePydantic, state: DagRunState | None, session: Session) -> pendulum.DateTime | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    The execution date from property previous_ti_success.\n\n    :param task_instance: the task instance\n    :param session: SQLAlchemy ORM Session\n    :param state: If passed, it only take into account instances of a specific state.\n\n    :meta private:\n    '
    log.debug('previous_execution_date was called')
    prev_ti = task_instance.get_previous_ti(state=state, session=session)
    return pendulum.instance(prev_ti.execution_date) if prev_ti and prev_ti.execution_date else None

def _email_alert(*, task_instance: TaskInstance | TaskInstancePydantic, exception, task: BaseOperator) -> None:
    if False:
        print('Hello World!')
    '\n    Send alert email with exception information.\n\n    :param task_instance: the task instance\n    :param exception: the exception\n    :param task: task related to the exception\n\n    :meta private:\n    '
    (subject, html_content, html_content_err) = task_instance.get_email_subject_content(exception, task=task)
    assert task.email
    try:
        send_email(task.email, subject, html_content)
    except Exception:
        send_email(task.email, subject, html_content_err)

def _get_email_subject_content(*, task_instance: TaskInstance | TaskInstancePydantic, exception: BaseException, task: BaseOperator | None=None) -> tuple[str, str, str]:
    if False:
        while True:
            i = 10
    '\n    Get the email subject content for exceptions.\n\n    :param task_instance: the task instance\n    :param exception: the exception sent in the email\n    :param task:\n\n    :meta private:\n    '
    if task is None:
        task = getattr(task_instance, 'task')
    use_default = task is None
    exception_html = str(exception).replace('\n', '<br>')
    default_subject = 'Airflow alert: {{ti}}'
    default_html_content = 'Try {{try_number}} out of {{max_tries + 1}}<br>Exception:<br>{{exception_html}}<br>Log: <a href="{{ti.log_url}}">Link</a><br>Host: {{ti.hostname}}<br>Mark success: <a href="{{ti.mark_success_url}}">Link</a><br>'
    default_html_content_err = 'Try {{try_number}} out of {{max_tries + 1}}<br>Exception:<br>Failed attempt to attach error logs<br>Log: <a href="{{ti.log_url}}">Link</a><br>Host: {{ti.hostname}}<br>Mark success: <a href="{{ti.mark_success_url}}">Link</a><br>'
    current_try_number = task_instance.try_number - 1
    additional_context: dict[str, Any] = {'exception': exception, 'exception_html': exception_html, 'try_number': current_try_number, 'max_tries': task_instance.max_tries}
    if use_default:
        default_context = {'ti': task_instance, **additional_context}
        jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)), autoescape=True)
        subject = jinja_env.from_string(default_subject).render(**default_context)
        html_content = jinja_env.from_string(default_html_content).render(**default_context)
        html_content_err = jinja_env.from_string(default_html_content_err).render(**default_context)
    else:
        dag = task_instance.task.get_dag()
        if dag:
            jinja_env = dag.get_template_env(force_sandboxed=True)
        else:
            jinja_env = SandboxedEnvironment(cache_size=0)
        jinja_context = task_instance.get_template_context()
        context_merge(jinja_context, additional_context)

        def render(key: str, content: str) -> str:
            if False:
                while True:
                    i = 10
            if conf.has_option('email', key):
                path = conf.get_mandatory_value('email', key)
                try:
                    with open(path) as f:
                        content = f.read()
                except FileNotFoundError:
                    log.warning("Could not find email template file '%s'. Using defaults...", path)
                except OSError:
                    log.exception('Error while using email template %s. Using defaults...', path)
            return render_template_to_string(jinja_env.from_string(content), jinja_context)
        subject = render('subject_template', default_subject)
        html_content = render('html_content_template', default_html_content)
        html_content_err = render('html_content_template', default_html_content_err)
    return (subject, html_content, html_content_err)

def _run_finished_callback(*, callbacks: None | TaskStateChangeCallback | list[TaskStateChangeCallback], context: Context) -> None:
    if False:
        return 10
    '\n    Run callback after task finishes.\n\n    :param callbacks: callbacks to run\n    :param context: callbacks context\n\n    :meta private:\n    '
    if callbacks:
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        for callback in callbacks:
            try:
                callback(context)
            except Exception:
                callback_name = qualname(callback).split('.')[-1]
                log.exception('Error when executing %s callback', callback_name)

def _log_state(*, task_instance: TaskInstance | TaskInstancePydantic, lead_msg: str='') -> None:
    if False:
        return 10
    '\n    Log task state.\n\n    :param task_instance: the task instance\n    :param lead_msg: lead message\n\n    :meta private:\n    '
    params = [lead_msg, str(task_instance.state).upper(), task_instance.dag_id, task_instance.task_id]
    message = '%sMarking task as %s. dag_id=%s, task_id=%s, '
    if task_instance.map_index >= 0:
        params.append(task_instance.map_index)
        message += 'map_index=%d, '
    log.info(message + 'execution_date=%s, start_date=%s, end_date=%s', *params, _date_or_empty(task_instance=task_instance, attr='execution_date'), _date_or_empty(task_instance=task_instance, attr='start_date'), _date_or_empty(task_instance=task_instance, attr='end_date'))

def _date_or_empty(*, task_instance: TaskInstance | TaskInstancePydantic, attr: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetch a date attribute or None of it does not exist.\n\n    :param task_instance: the task instance\n    :param attr: the attribute name\n\n    :meta private:\n    '
    result: datetime | None = getattr(task_instance, attr, None)
    return result.strftime('%Y%m%dT%H%M%S') if result else ''

def _get_previous_ti(*, task_instance: TaskInstance | TaskInstancePydantic, session: Session, state: DagRunState | None=None) -> TaskInstance | TaskInstancePydantic | None:
    if False:
        while True:
            i = 10
    '\n    The task instance for the task that ran before this task instance.\n\n    :param task_instance: the task instance\n    :param state: If passed, it only take into account instances of a specific state.\n    :param session: SQLAlchemy ORM Session\n\n    :meta private:\n    '
    dagrun = task_instance.get_previous_dagrun(state, session=session)
    if dagrun is None:
        return None
    return dagrun.get_task_instance(task_instance.task_id, session=session)

class TaskInstance(Base, LoggingMixin):
    """
    Task instances store the state of a task instance.

    This table is the authority and single source of truth around what tasks
    have run and the state they are in.

    The SqlAlchemy model doesn't have a SqlAlchemy foreign key to the task or
    dag model deliberately to have more control over transactions.

    Database transactions on this table should insure double triggers and
    any confusion around what task instances are or aren't ready to run
    even while multiple schedulers may be firing task instances.

    A value of -1 in map_index represents any of: a TI without mapped tasks;
    a TI with mapped tasks that has yet to be expanded (state=pending);
    a TI with mapped tasks that expanded to an empty list (state=skipped).
    """
    __tablename__ = 'task_instance'
    task_id = Column(StringID(), primary_key=True, nullable=False)
    dag_id = Column(StringID(), primary_key=True, nullable=False)
    run_id = Column(StringID(), primary_key=True, nullable=False)
    map_index = Column(Integer, primary_key=True, nullable=False, server_default=text('-1'))
    start_date = Column(UtcDateTime)
    end_date = Column(UtcDateTime)
    duration = Column(Float)
    state = Column(String(20))
    _try_number = Column('try_number', Integer, default=0)
    max_tries = Column(Integer, server_default=text('-1'))
    hostname = Column(String(1000))
    unixname = Column(String(1000))
    job_id = Column(Integer)
    pool = Column(String(256), nullable=False)
    pool_slots = Column(Integer, default=1, nullable=False)
    queue = Column(String(256))
    priority_weight = Column(Integer)
    operator = Column(String(1000))
    custom_operator_name = Column(String(1000))
    queued_dttm = Column(UtcDateTime)
    queued_by_job_id = Column(Integer)
    pid = Column(Integer)
    executor_config = Column(ExecutorConfigType(pickler=dill))
    updated_at = Column(UtcDateTime, default=timezone.utcnow, onupdate=timezone.utcnow)
    external_executor_id = Column(StringID())
    trigger_id = Column(Integer)
    trigger_timeout = Column(DateTime)
    next_method = Column(String(1000))
    next_kwargs = Column(MutableDict.as_mutable(ExtendedJSON))
    __table_args__ = (Index('ti_dag_state', dag_id, state), Index('ti_dag_run', dag_id, run_id), Index('ti_state', state), Index('ti_state_lkp', dag_id, task_id, run_id, state), Index('ti_state_incl_start_date', dag_id, task_id, state, postgresql_include=['start_date']), Index('ti_pool', pool, state, priority_weight), Index('ti_job_id', job_id), Index('ti_trigger_id', trigger_id), PrimaryKeyConstraint('dag_id', 'task_id', 'run_id', 'map_index', name='task_instance_pkey', mssql_clustered=True), ForeignKeyConstraint([trigger_id], ['trigger.id'], name='task_instance_trigger_id_fkey', ondelete='CASCADE'), ForeignKeyConstraint([dag_id, run_id], ['dag_run.dag_id', 'dag_run.run_id'], name='task_instance_dag_run_fkey', ondelete='CASCADE'))
    dag_model: DagModel = relationship('DagModel', primaryjoin='TaskInstance.dag_id == DagModel.dag_id', foreign_keys=dag_id, uselist=False, innerjoin=True, viewonly=True)
    trigger = relationship('Trigger', uselist=False, back_populates='task_instance')
    triggerer_job = association_proxy('trigger', 'triggerer_job')
    dag_run = relationship('DagRun', back_populates='task_instances', lazy='joined', innerjoin=True)
    rendered_task_instance_fields = relationship('RenderedTaskInstanceFields', lazy='noload', uselist=False)
    execution_date = association_proxy('dag_run', 'execution_date')
    task_instance_note = relationship('TaskInstanceNote', back_populates='task_instance', uselist=False, cascade='all, delete, delete-orphan')
    note = association_proxy('task_instance_note', 'content', creator=_creator_note)
    task: Operator
    test_mode: bool = False
    is_trigger_log_context: bool = False
    'Indicate to FileTaskHandler that logging context should be set up for trigger logging.\n\n    :meta private:\n    '

    def __init__(self, task: Operator, execution_date: datetime | None=None, run_id: str | None=None, state: str | None=None, map_index: int=-1):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dag_id = task.dag_id
        self.task_id = task.task_id
        self.map_index = map_index
        self.refresh_from_task(task)
        self.init_on_load()
        if run_id is None and execution_date is not None:
            from airflow.models.dagrun import DagRun
            warnings.warn('Passing an execution_date to `TaskInstance()` is deprecated in favour of passing a run_id', RemovedInAirflow3Warning, stacklevel=4)
            if execution_date and (not timezone.is_localized(execution_date)):
                self.log.warning('execution date %s has no timezone information. Using default from dag or system', execution_date)
                if self.task.has_dag():
                    if TYPE_CHECKING:
                        assert self.task.dag
                    execution_date = timezone.make_aware(execution_date, self.task.dag.timezone)
                else:
                    execution_date = timezone.make_aware(execution_date)
                execution_date = timezone.convert_to_utc(execution_date)
            with create_session() as session:
                run_id = session.query(DagRun.run_id).filter_by(dag_id=self.dag_id, execution_date=execution_date).scalar()
                if run_id is None:
                    raise DagRunNotFound(f'DagRun for {self.dag_id!r} with date {execution_date} not found') from None
        self.run_id = run_id
        self.try_number = 0
        self.max_tries = self.task.retries
        self.unixname = getuser()
        if state:
            self.state = state
        self.hostname = ''
        self.raw = False
        self.test_mode = False

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.task_id, self.dag_id, self.run_id, self.map_index))

    @property
    def stats_tags(self) -> dict[str, str]:
        if False:
            return 10
        'Returns task instance tags.'
        return _stats_tags(task_instance=self)

    @staticmethod
    def insert_mapping(run_id: str, task: Operator, map_index: int) -> dict[str, Any]:
        if False:
            return 10
        'Insert mapping.\n\n        :meta private:\n        '
        return {'dag_id': task.dag_id, 'task_id': task.task_id, 'run_id': run_id, '_try_number': 0, 'hostname': '', 'unixname': getuser(), 'queue': task.queue, 'pool': task.pool, 'pool_slots': task.pool_slots, 'priority_weight': task.priority_weight_total, 'run_as_user': task.run_as_user, 'max_tries': task.retries, 'executor_config': task.executor_config, 'operator': task.task_type, 'custom_operator_name': getattr(task, 'custom_operator_name', None), 'map_index': map_index}

    @reconstructor
    def init_on_load(self) -> None:
        if False:
            print('Hello World!')
        "Initialize the attributes that aren't stored in the DB."
        self._log = logging.getLogger('airflow.task')
        self.test_mode = False

    @hybrid_property
    def try_number(self):
        if False:
            print('Hello World!')
        '\n        Return the try number that a task number will be when it is actually run.\n\n        If the TaskInstance is currently running, this will match the column in the\n        database, in all other cases this will be incremented.\n\n        This is designed so that task logs end up in the right file.\n        '
        return _get_try_number(task_instance=self)

    @try_number.setter
    def try_number(self, value: int) -> None:
        if False:
            print('Hello World!')
        '\n        Set a task try number.\n\n        :param value: the try number\n        '
        _set_try_number(task_instance=self, value=value)

    @property
    def prev_attempted_tries(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Calculate the number of previously attempted tries, defaulting to 0.\n\n        Expose this for the Task Tries and Gantt graph views.\n        Using `try_number` throws off the counts for non-running tasks.\n        Also useful in error logging contexts to get the try number for the last try that was attempted.\n        https://issues.apache.org/jira/browse/AIRFLOW-2143\n        '
        return self._try_number

    @property
    def next_try_number(self) -> int:
        if False:
            print('Hello World!')
        return self._try_number + 1

    @property
    def operator_name(self) -> str | None:
        if False:
            return 10
        '@property: use a more friendly display name for the operator, if set.'
        return self.custom_operator_name or self.operator

    @staticmethod
    def _command_as_list(ti: TaskInstance | TaskInstancePydantic, mark_success: bool=False, ignore_all_deps: bool=False, ignore_task_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_ti_state: bool=False, local: bool=False, pickle_id: int | None=None, raw: bool=False, job_id: str | None=None, pool: str | None=None, cfg_path: str | None=None) -> list[str]:
        if False:
            while True:
                i = 10
        dag: DAG | DagModel | DagModelPydantic | None
        if hasattr(ti, 'task') and hasattr(ti.task, 'dag') and (ti.task.dag is not None):
            dag = ti.task.dag
        else:
            dag = ti.dag_model
        if dag is None:
            raise ValueError('DagModel is empty')
        should_pass_filepath = not pickle_id and dag
        path: PurePath | None = None
        if should_pass_filepath:
            if dag.is_subdag:
                if TYPE_CHECKING:
                    assert dag.parent_dag is not None
                path = dag.parent_dag.relative_fileloc
            else:
                path = dag.relative_fileloc
            if path:
                if not path.is_absolute():
                    path = 'DAGS_FOLDER' / path
        return TaskInstance.generate_command(ti.dag_id, ti.task_id, run_id=ti.run_id, mark_success=mark_success, ignore_all_deps=ignore_all_deps, ignore_task_deps=ignore_task_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_ti_state=ignore_ti_state, local=local, pickle_id=pickle_id, file_path=path, raw=raw, job_id=job_id, pool=pool, cfg_path=cfg_path, map_index=ti.map_index)

    def command_as_list(self, mark_success: bool=False, ignore_all_deps: bool=False, ignore_task_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_ti_state: bool=False, local: bool=False, pickle_id: int | None=None, raw: bool=False, job_id: str | None=None, pool: str | None=None, cfg_path: str | None=None) -> list[str]:
        if False:
            i = 10
            return i + 15
        '\n        Return a command that can be executed anywhere where airflow is installed.\n\n        This command is part of the message sent to executors by the orchestrator.\n        '
        return TaskInstance._command_as_list(ti=self, mark_success=mark_success, ignore_all_deps=ignore_all_deps, ignore_task_deps=ignore_task_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_ti_state=ignore_ti_state, local=local, pickle_id=pickle_id, raw=raw, job_id=job_id, pool=pool, cfg_path=cfg_path)

    @staticmethod
    def generate_command(dag_id: str, task_id: str, run_id: str, mark_success: bool=False, ignore_all_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_task_deps: bool=False, ignore_ti_state: bool=False, local: bool=False, pickle_id: int | None=None, file_path: PurePath | str | None=None, raw: bool=False, job_id: str | None=None, pool: str | None=None, cfg_path: str | None=None, map_index: int=-1) -> list[str]:
        if False:
            print('Hello World!')
        "\n        Generate the shell command required to execute this task instance.\n\n        :param dag_id: DAG ID\n        :param task_id: Task ID\n        :param run_id: The run_id of this task's DagRun\n        :param mark_success: Whether to mark the task as successful\n        :param ignore_all_deps: Ignore all ignorable dependencies.\n            Overrides the other ignore_* parameters.\n        :param ignore_depends_on_past: Ignore depends_on_past parameter of DAGs\n            (e.g. for Backfills)\n        :param wait_for_past_depends_before_skipping: Wait for past depends before marking the ti as skipped\n        :param ignore_task_deps: Ignore task-specific dependencies such as depends_on_past\n            and trigger rule\n        :param ignore_ti_state: Ignore the task instance's previous failure/success\n        :param local: Whether to run the task locally\n        :param pickle_id: If the DAG was serialized to the DB, the ID\n            associated with the pickled DAG\n        :param file_path: path to the file containing the DAG definition\n        :param raw: raw mode (needs more details)\n        :param job_id: job ID (needs more details)\n        :param pool: the Airflow pool that the task should run in\n        :param cfg_path: the Path to the configuration file\n        :return: shell command that can be used to run the task instance\n        "
        cmd = ['airflow', 'tasks', 'run', dag_id, task_id, run_id]
        if mark_success:
            cmd.extend(['--mark-success'])
        if pickle_id:
            cmd.extend(['--pickle', str(pickle_id)])
        if job_id:
            cmd.extend(['--job-id', str(job_id)])
        if ignore_all_deps:
            cmd.extend(['--ignore-all-dependencies'])
        if ignore_task_deps:
            cmd.extend(['--ignore-dependencies'])
        if ignore_depends_on_past:
            cmd.extend(['--depends-on-past', 'ignore'])
        elif wait_for_past_depends_before_skipping:
            cmd.extend(['--depends-on-past', 'wait'])
        if ignore_ti_state:
            cmd.extend(['--force'])
        if local:
            cmd.extend(['--local'])
        if pool:
            cmd.extend(['--pool', pool])
        if raw:
            cmd.extend(['--raw'])
        if file_path:
            cmd.extend(['--subdir', os.fspath(file_path)])
        if cfg_path:
            cmd.extend(['--cfg-path', cfg_path])
        if map_index != -1:
            cmd.extend(['--map-index', str(map_index)])
        return cmd

    @property
    def log_url(self) -> str:
        if False:
            while True:
                i = 10
        'Log URL for TaskInstance.'
        iso = quote(self.execution_date.isoformat())
        base_url = conf.get_mandatory_value('webserver', 'BASE_URL')
        return f'{base_url}/log?execution_date={iso}&task_id={self.task_id}&dag_id={self.dag_id}&map_index={self.map_index}'

    @property
    def mark_success_url(self) -> str:
        if False:
            i = 10
            return i + 15
        'URL to mark TI success.'
        base_url = conf.get_mandatory_value('webserver', 'BASE_URL')
        return f'{base_url}/confirm?task_id={self.task_id}&dag_id={self.dag_id}&dag_run_id={quote(self.run_id)}&upstream=false&downstream=false&state=success'

    @provide_session
    def current_state(self, session: Session=NEW_SESSION) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the very latest state from the database.\n\n        If a session is passed, we use and looking up the state becomes part of the session,\n        otherwise a new session is used.\n\n        sqlalchemy.inspect is used here to get the primary keys ensuring that if they change\n        it will not regress\n\n        :param session: SQLAlchemy ORM Session\n        '
        filters = (col == getattr(self, col.name) for col in inspect(TaskInstance).primary_key)
        return session.query(TaskInstance.state).filter(*filters).scalar()

    @provide_session
    def error(self, session: Session=NEW_SESSION) -> None:
        if False:
            while True:
                i = 10
        "\n        Force the task instance's state to FAILED in the database.\n\n        :param session: SQLAlchemy ORM Session\n        "
        self.log.error('Recording the task instance as FAILED')
        self.state = TaskInstanceState.FAILED
        session.merge(self)
        session.commit()

    @classmethod
    @internal_api_call
    @provide_session
    def get_task_instance(cls, dag_id: str, run_id: str, task_id: str, map_index: int, select_columns: bool=False, lock_for_update: bool=False, session: Session=NEW_SESSION) -> TaskInstance | TaskInstancePydantic | None:
        if False:
            print('Hello World!')
        query = session.query(*TaskInstance.__table__.columns) if select_columns else session.query(TaskInstance)
        query = query.filter_by(dag_id=dag_id, run_id=run_id, task_id=task_id, map_index=map_index)
        if lock_for_update:
            for attempt in run_with_db_retries(logger=cls.logger()):
                with attempt:
                    return query.with_for_update().one_or_none()
        else:
            return query.one_or_none()
        return None

    @provide_session
    def refresh_from_db(self, session: Session=NEW_SESSION, lock_for_update: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Refresh the task instance from the database based on the primary key.\n\n        :param session: SQLAlchemy ORM Session\n        :param lock_for_update: if True, indicates that the database should\n            lock the TaskInstance (issuing a FOR UPDATE clause) until the\n            session is committed.\n        '
        _refresh_from_db(task_instance=self, session=session, lock_for_update=lock_for_update)

    def refresh_from_task(self, task: Operator, pool_override: str | None=None) -> None:
        if False:
            return 10
        "\n        Copy common attributes from the given task.\n\n        :param task: The task object to copy from\n        :param pool_override: Use the pool_override instead of task's pool\n        "
        _refresh_from_task(task_instance=self, task=task, pool_override=pool_override)

    @provide_session
    def clear_xcom_data(self, session: Session=NEW_SESSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear all XCom data from the database for the task instance.\n\n        If the task is unmapped, all XComs matching this task ID in the same DAG\n        run are removed. If the task is mapped, only the one with matching map\n        index is removed.\n\n        :param session: SQLAlchemy ORM Session\n        '
        self.log.debug('Clearing XCom data')
        if self.map_index < 0:
            map_index: int | None = None
        else:
            map_index = self.map_index
        XCom.clear(dag_id=self.dag_id, task_id=self.task_id, run_id=self.run_id, map_index=map_index, session=session)

    @property
    def key(self) -> TaskInstanceKey:
        if False:
            i = 10
            return i + 15
        'Returns a tuple that identifies the task instance uniquely.'
        return TaskInstanceKey(self.dag_id, self.task_id, self.run_id, self.try_number, self.map_index)

    @provide_session
    def set_state(self, state: str | None, session: Session=NEW_SESSION) -> bool:
        if False:
            print('Hello World!')
        '\n        Set TaskInstance state.\n\n        :param state: State to set for the TI\n        :param session: SQLAlchemy ORM Session\n        :return: Was the state changed\n        '
        if self.state == state:
            return False
        current_time = timezone.utcnow()
        self.log.debug('Setting task state for %s to %s', self, state)
        self.state = state
        self.start_date = self.start_date or current_time
        if self.state in State.finished or self.state == TaskInstanceState.UP_FOR_RETRY:
            self.end_date = self.end_date or current_time
            self.duration = (self.end_date - self.start_date).total_seconds()
        session.merge(self)
        return True

    @property
    def is_premature(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns whether a task is in UP_FOR_RETRY state and its retry interval has elapsed.'
        return self.state == TaskInstanceState.UP_FOR_RETRY and (not self.ready_for_retry())

    @provide_session
    def are_dependents_done(self, session: Session=NEW_SESSION) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Check whether the immediate dependents of this task instance have succeeded or have been skipped.\n\n        This is meant to be used by wait_for_downstream.\n\n        This is useful when you do not want to start processing the next\n        schedule of a task until the dependents are done. For instance,\n        if the task DROPs and recreates a table.\n\n        :param session: SQLAlchemy ORM Session\n        '
        task = self.task
        if not task.downstream_task_ids:
            return True
        ti = session.query(func.count(TaskInstance.task_id)).filter(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id.in_(task.downstream_task_ids), TaskInstance.run_id == self.run_id, TaskInstance.state.in_((TaskInstanceState.SKIPPED, TaskInstanceState.SUCCESS)))
        count = ti[0][0]
        return count == len(task.downstream_task_ids)

    @provide_session
    def get_previous_dagrun(self, state: DagRunState | None=None, session: Session | None=None) -> DagRun | None:
        if False:
            print('Hello World!')
        "\n        Return the DagRun that ran before this task instance's DagRun.\n\n        :param state: If passed, it only take into account instances of a specific state.\n        :param session: SQLAlchemy ORM Session.\n        "
        return _get_previous_dagrun(task_instance=self, state=state, session=session)

    @provide_session
    def get_previous_ti(self, state: DagRunState | None=None, session: Session=NEW_SESSION) -> TaskInstance | TaskInstancePydantic | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the task instance for the task that ran before this task instance.\n\n        :param session: SQLAlchemy ORM Session\n        :param state: If passed, it only take into account instances of a specific state.\n        '
        return _get_previous_ti(task_instance=self, state=state, session=session)

    @property
    def previous_ti(self) -> TaskInstance | TaskInstancePydantic | None:
        if False:
            while True:
                i = 10
        '\n        This attribute is deprecated.\n\n        Please use :class:`airflow.models.taskinstance.TaskInstance.get_previous_ti`.\n        '
        warnings.warn('\n            This attribute is deprecated.\n            Please use `airflow.models.taskinstance.TaskInstance.get_previous_ti` method.\n            ', RemovedInAirflow3Warning, stacklevel=2)
        return self.get_previous_ti()

    @property
    def previous_ti_success(self) -> TaskInstance | TaskInstancePydantic | None:
        if False:
            while True:
                i = 10
        '\n        This attribute is deprecated.\n\n        Please use :class:`airflow.models.taskinstance.TaskInstance.get_previous_ti`.\n        '
        warnings.warn('\n            This attribute is deprecated.\n            Please use `airflow.models.taskinstance.TaskInstance.get_previous_ti` method.\n            ', RemovedInAirflow3Warning, stacklevel=2)
        return self.get_previous_ti(state=DagRunState.SUCCESS)

    @provide_session
    def get_previous_execution_date(self, state: DagRunState | None=None, session: Session=NEW_SESSION) -> pendulum.DateTime | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the execution date from property previous_ti_success.\n\n        :param state: If passed, it only take into account instances of a specific state.\n        :param session: SQLAlchemy ORM Session\n        '
        return _get_previous_execution_date(task_instance=self, state=state, session=session)

    @provide_session
    def get_previous_start_date(self, state: DagRunState | None=None, session: Session=NEW_SESSION) -> pendulum.DateTime | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the start date from property previous_ti_success.\n\n        :param state: If passed, it only take into account instances of a specific state.\n        :param session: SQLAlchemy ORM Session\n        '
        self.log.debug('previous_start_date was called')
        prev_ti = self.get_previous_ti(state=state, session=session)
        return pendulum.instance(prev_ti.start_date) if prev_ti and prev_ti.start_date else None

    @property
    def previous_start_date_success(self) -> pendulum.DateTime | None:
        if False:
            i = 10
            return i + 15
        '\n        This attribute is deprecated.\n\n        Please use :class:`airflow.models.taskinstance.TaskInstance.get_previous_start_date`.\n        '
        warnings.warn('\n            This attribute is deprecated.\n            Please use `airflow.models.taskinstance.TaskInstance.get_previous_start_date` method.\n            ', RemovedInAirflow3Warning, stacklevel=2)
        return self.get_previous_start_date(state=DagRunState.SUCCESS)

    @provide_session
    def are_dependencies_met(self, dep_context: DepContext | None=None, session: Session=NEW_SESSION, verbose: bool=False) -> bool:
        if False:
            while True:
                i = 10
        '\n        Are all conditions met for this task instance to be run given the context for the dependencies.\n\n        (e.g. a task instance being force run from the UI will ignore some dependencies).\n\n        :param dep_context: The execution context that determines the dependencies that should be evaluated.\n        :param session: database session\n        :param verbose: whether log details on failed dependencies on info or debug log level\n        '
        dep_context = dep_context or DepContext()
        failed = False
        verbose_aware_logger = self.log.info if verbose else self.log.debug
        for dep_status in self.get_failed_dep_statuses(dep_context=dep_context, session=session):
            failed = True
            verbose_aware_logger("Dependencies not met for %s, dependency '%s' FAILED: %s", self, dep_status.dep_name, dep_status.reason)
        if failed:
            return False
        verbose_aware_logger('Dependencies all met for dep_context=%s ti=%s', dep_context.description, self)
        return True

    @provide_session
    def get_failed_dep_statuses(self, dep_context: DepContext | None=None, session: Session=NEW_SESSION):
        if False:
            return 10
        'Get failed Dependencies.'
        dep_context = dep_context or DepContext()
        for dep in dep_context.deps | self.task.deps:
            for dep_status in dep.get_dep_statuses(self, session, dep_context):
                self.log.debug("%s dependency '%s' PASSED: %s, %s", self, dep_status.dep_name, dep_status.passed, dep_status.reason)
                if not dep_status.passed:
                    yield dep_status

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        prefix = f'<TaskInstance: {self.dag_id}.{self.task_id} {self.run_id} '
        if self.map_index != -1:
            prefix += f'map_index={self.map_index} '
        return prefix + f'[{self.state}]>'

    def next_retry_datetime(self):
        if False:
            while True:
                i = 10
        '\n        Get datetime of the next retry if the task instance fails.\n\n        For exponential backoff, retry_delay is used as base and will be converted to seconds.\n        '
        from airflow.models.abstractoperator import MAX_RETRY_DELAY
        delay = self.task.retry_delay
        if self.task.retry_exponential_backoff:
            min_backoff = math.ceil(delay.total_seconds() * 2 ** (self.try_number - 2))
            if min_backoff < 1:
                min_backoff = 1
            ti_hash = int(hashlib.sha1(f'{self.dag_id}#{self.task_id}#{self.execution_date}#{self.try_number}'.encode()).hexdigest(), 16)
            modded_hash = min_backoff + ti_hash % min_backoff
            delay_backoff_in_seconds = min(modded_hash, MAX_RETRY_DELAY)
            delay = timedelta(seconds=delay_backoff_in_seconds)
            if self.task.max_retry_delay:
                delay = min(self.task.max_retry_delay, delay)
        return self.end_date + delay

    def ready_for_retry(self) -> bool:
        if False:
            print('Hello World!')
        'Check on whether the task instance is in the right state and timeframe to be retried.'
        return self.state == TaskInstanceState.UP_FOR_RETRY and self.next_retry_datetime() < timezone.utcnow()

    @provide_session
    def get_dagrun(self, session: Session=NEW_SESSION) -> DagRun:
        if False:
            print('Hello World!')
        '\n        Return the DagRun for this TaskInstance.\n\n        :param session: SQLAlchemy ORM Session\n        :return: DagRun\n        '
        info = inspect(self)
        if info.attrs.dag_run.loaded_value is not NO_VALUE:
            if hasattr(self, 'task'):
                self.dag_run.dag = self.task.dag
            return self.dag_run
        from airflow.models.dagrun import DagRun
        dr = session.query(DagRun).filter(DagRun.dag_id == self.dag_id, DagRun.run_id == self.run_id).one()
        if hasattr(self, 'task'):
            dr.dag = self.task.dag
        set_committed_value(self, 'dag_run', dr)
        return dr

    @classmethod
    @internal_api_call
    @provide_session
    def _check_and_change_state_before_execution(cls, task_instance: TaskInstance | TaskInstancePydantic, verbose: bool=True, ignore_all_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_task_deps: bool=False, ignore_ti_state: bool=False, mark_success: bool=False, test_mode: bool=False, hostname: str='', job_id: str | None=None, pool: str | None=None, external_executor_id: str | None=None, session: Session=NEW_SESSION) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        Check dependencies and then sets state to RUNNING if they are met.\n\n        Returns True if and only if state is set to RUNNING, which implies that task should be\n        executed, in preparation for _run_raw_task.\n\n        :param verbose: whether to turn on more verbose logging\n        :param ignore_all_deps: Ignore all of the non-critical dependencies, just runs\n        :param ignore_depends_on_past: Ignore depends_on_past DAG attribute\n        :param wait_for_past_depends_before_skipping: Wait for past depends before mark the ti as skipped\n        :param ignore_task_deps: Don't check the dependencies of this TaskInstance's task\n        :param ignore_ti_state: Disregards previous task instance state\n        :param mark_success: Don't run the task, mark its state as success\n        :param test_mode: Doesn't record success or failure in the DB\n        :param hostname: The hostname of the worker running the task instance.\n        :param job_id: Job (BackfillJob / LocalTaskJob / SchedulerJob) ID\n        :param pool: specifies the pool to use to run the task instance\n        :param external_executor_id: The identifier of the celery executor\n        :param session: SQLAlchemy ORM Session\n        :return: whether the state was changed to running or not\n        "
        if isinstance(task_instance, TaskInstance):
            ti: TaskInstance = task_instance
        else:
            filters = (col == getattr(task_instance, col.name) for col in inspect(TaskInstance).primary_key)
            ti = session.query(TaskInstance).filter(*filters).scalar()
        task = task_instance.task
        ti.refresh_from_task(task, pool_override=pool)
        ti.test_mode = test_mode
        ti.refresh_from_db(session=session, lock_for_update=True)
        ti.job_id = job_id
        ti.hostname = hostname
        ti.pid = None
        if not ignore_all_deps and (not ignore_ti_state) and (ti.state == TaskInstanceState.SUCCESS):
            Stats.incr('previously_succeeded', tags=ti.stats_tags)
        if not mark_success:
            non_requeueable_dep_context = DepContext(deps=RUNNING_DEPS - REQUEUEABLE_DEPS, ignore_all_deps=ignore_all_deps, ignore_ti_state=ignore_ti_state, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_task_deps=ignore_task_deps, description='non-requeueable deps')
            if not ti.are_dependencies_met(dep_context=non_requeueable_dep_context, session=session, verbose=True):
                session.commit()
                return False
            ti.start_date = ti.start_date if ti.next_method else timezone.utcnow()
            if ti.state == TaskInstanceState.UP_FOR_RESCHEDULE:
                tr_start_date = session.scalar(TR.stmt_for_task_instance(ti, descending=False).with_only_columns(TR.start_date).limit(1))
                if tr_start_date:
                    ti.start_date = tr_start_date
            dep_context = DepContext(deps=REQUEUEABLE_DEPS, ignore_all_deps=ignore_all_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_task_deps=ignore_task_deps, ignore_ti_state=ignore_ti_state, description='requeueable deps')
            if not ti.are_dependencies_met(dep_context=dep_context, session=session, verbose=True):
                ti.state = None
                cls.logger().warning('Rescheduling due to concurrency limits reached at task runtime. Attempt %s of %s. State set to NONE.', ti.try_number, ti.max_tries + 1)
                ti.queued_dttm = timezone.utcnow()
                session.merge(ti)
                session.commit()
                return False
        if ti.next_kwargs is not None:
            cls.logger().info('Resuming after deferral')
        else:
            cls.logger().info('Starting attempt %s of %s', ti.try_number, ti.max_tries + 1)
        ti._try_number += 1
        if not test_mode:
            session.add(Log(TaskInstanceState.RUNNING.value, ti))
        ti.state = TaskInstanceState.RUNNING
        ti.emit_state_change_metric(TaskInstanceState.RUNNING)
        ti.external_executor_id = external_executor_id
        ti.end_date = None
        if not test_mode:
            session.merge(ti).task = task
        session.commit()
        settings.engine.dispose()
        if verbose:
            if mark_success:
                cls.logger().info('Marking success for %s on %s', ti.task, ti.execution_date)
            else:
                cls.logger().info('Executing %s on %s', ti.task, ti.execution_date)
        return True

    @provide_session
    def check_and_change_state_before_execution(self, verbose: bool=True, ignore_all_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_task_deps: bool=False, ignore_ti_state: bool=False, mark_success: bool=False, test_mode: bool=False, job_id: str | None=None, pool: str | None=None, external_executor_id: str | None=None, session: Session=NEW_SESSION) -> bool:
        if False:
            print('Hello World!')
        return TaskInstance._check_and_change_state_before_execution(task_instance=self, verbose=verbose, ignore_all_deps=ignore_all_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_task_deps=ignore_task_deps, ignore_ti_state=ignore_ti_state, mark_success=mark_success, test_mode=test_mode, hostname=get_hostname(), job_id=job_id, pool=pool, external_executor_id=external_executor_id, session=session)

    def emit_state_change_metric(self, new_state: TaskInstanceState) -> None:
        if False:
            print('Hello World!')
        '\n        Send a time metric representing how much time a given state transition took.\n\n        The previous state and metric name is deduced from the state the task was put in.\n\n        :param new_state: The state that has just been set for this task.\n            We do not use `self.state`, because sometimes the state is updated directly in the DB and not in\n            the local TaskInstance object.\n            Supported states: QUEUED and RUNNING\n        '
        if self.end_date:
            return
        if new_state == TaskInstanceState.RUNNING:
            metric_name = 'queued_duration'
            if self.queued_dttm is None:
                self.log.warning('cannot record %s for task %s because previous state change time has not been saved', metric_name, self.task_id)
                return
            timing = (timezone.utcnow() - self.queued_dttm).total_seconds()
        elif new_state == TaskInstanceState.QUEUED:
            metric_name = 'scheduled_duration'
            if self.start_date is None:
                self.log.warning('cannot record %s for task %s because previous state change time has not been saved', metric_name, self.task_id)
                return
            timing = (timezone.utcnow() - self.start_date).total_seconds()
        else:
            raise NotImplementedError('no metric emission setup for state %s', new_state)
        Stats.timing(f'dag.{self.dag_id}.{self.task_id}.{metric_name}', timing)
        Stats.timing(f'task.{metric_name}', timing, tags={'task_id': self.task_id, 'dag_id': self.dag_id})

    def clear_next_method_args(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Ensure we unset next_method and next_kwargs to ensure that any retries don't reuse them."
        _clear_next_method_args(task_instance=self)

    @provide_session
    @Sentry.enrich_errors
    def _run_raw_task(self, mark_success: bool=False, test_mode: bool=False, job_id: str | None=None, pool: str | None=None, session: Session=NEW_SESSION) -> TaskReturnCode | None:
        if False:
            return 10
        "\n        Run a task, update the state upon completion, and run any appropriate callbacks.\n\n        Immediately runs the task (without checking or changing db state\n        before execution) and then sets the appropriate final state after\n        completion and runs any post-execute callbacks. Meant to be called\n        only after another function changes the state to running.\n\n        :param mark_success: Don't run the task, mark its state as success\n        :param test_mode: Doesn't record success or failure in the DB\n        :param pool: specifies the pool to use to run the task instance\n        :param session: SQLAlchemy ORM Session\n        "
        self.test_mode = test_mode
        self.refresh_from_task(self.task, pool_override=pool)
        self.refresh_from_db(session=session)
        self.job_id = job_id
        self.hostname = get_hostname()
        self.pid = os.getpid()
        if not test_mode:
            session.merge(self)
            session.commit()
        actual_start_date = timezone.utcnow()
        Stats.incr(f'ti.start.{self.task.dag_id}.{self.task.task_id}', tags=self.stats_tags)
        Stats.incr('ti.start', tags=self.stats_tags)
        for state in State.task_states:
            Stats.incr(f'ti.finish.{self.task.dag_id}.{self.task.task_id}.{state}', count=0, tags=self.stats_tags)
            Stats.incr('ti.finish', count=0, tags={**self.stats_tags, 'state': str(state)})
        with set_current_task_instance_session(session=session):
            self.task = self.task.prepare_for_execution()
            context = self.get_template_context(ignore_param_exceptions=False)
            try:
                if not mark_success:
                    self._execute_task_with_callbacks(context, test_mode, session=session)
                if not test_mode:
                    self.refresh_from_db(lock_for_update=True, session=session)
                self.state = TaskInstanceState.SUCCESS
            except TaskDeferred as defer:
                self._defer_task(defer=defer, session=session)
                self.log.info('Pausing task as DEFERRED. dag_id=%s, task_id=%s, execution_date=%s, start_date=%s', self.dag_id, self.task_id, _date_or_empty(task_instance=self, attr='execution_date'), _date_or_empty(task_instance=self, attr='start_date'))
                if not test_mode:
                    session.add(Log(self.state, self))
                    session.merge(self)
                    session.commit()
                return TaskReturnCode.DEFERRED
            except AirflowSkipException as e:
                if e.args:
                    self.log.info(e)
                if not test_mode:
                    self.refresh_from_db(lock_for_update=True, session=session)
                self.state = TaskInstanceState.SKIPPED
            except AirflowRescheduleException as reschedule_exception:
                self._handle_reschedule(actual_start_date, reschedule_exception, test_mode, session=session)
                session.commit()
                return None
            except (AirflowFailException, AirflowSensorTimeout) as e:
                self.handle_failure(e, test_mode, context, force_fail=True, session=session)
                session.commit()
                raise
            except AirflowException as e:
                if not test_mode:
                    self.refresh_from_db(lock_for_update=True, session=session)
                if self.state in State.finished:
                    self.clear_next_method_args()
                    session.merge(self)
                    session.commit()
                    return None
                else:
                    self.handle_failure(e, test_mode, context, session=session)
                    session.commit()
                    raise
            except (Exception, KeyboardInterrupt) as e:
                self.handle_failure(e, test_mode, context, session=session)
                session.commit()
                raise
            finally:
                Stats.incr(f'ti.finish.{self.dag_id}.{self.task_id}.{self.state}', tags=self.stats_tags)
                Stats.incr('ti.finish', tags={**self.stats_tags, 'state': str(self.state)})
            self.clear_next_method_args()
            self.end_date = timezone.utcnow()
            _log_state(task_instance=self)
            self.set_duration()
            _run_finished_callback(callbacks=self.task.on_success_callback, context=context)
            if not test_mode:
                session.add(Log(self.state, self))
                session.merge(self).task = self.task
                if self.state == TaskInstanceState.SUCCESS:
                    self._register_dataset_changes(session=session)
                session.commit()
                if self.state == TaskInstanceState.SUCCESS:
                    get_listener_manager().hook.on_task_instance_success(previous_state=TaskInstanceState.RUNNING, task_instance=self, session=session)
            return None

    def _register_dataset_changes(self, *, session: Session) -> None:
        if False:
            for i in range(10):
                print('nop')
        for obj in self.task.outlets or []:
            self.log.debug('outlet obj %s', obj)
            if isinstance(obj, Dataset):
                dataset_manager.register_dataset_change(task_instance=self, dataset=obj, session=session)

    def _execute_task_with_callbacks(self, context, test_mode: bool=False, *, session: Session):
        if False:
            return 10
        'Prepare Task for Execution.'
        from airflow.models.renderedtifields import RenderedTaskInstanceFields
        parent_pid = os.getpid()

        def signal_handler(signum, frame):
            if False:
                while True:
                    i = 10
            pid = os.getpid()
            if pid != parent_pid:
                os._exit(1)
                return
            self.log.error('Received SIGTERM. Terminating subprocesses.')
            self.task.on_kill()
            raise AirflowException('Task received SIGTERM signal')
        signal.signal(signal.SIGTERM, signal_handler)
        if not self.next_method:
            self.clear_xcom_data()
        with Stats.timer(f'dag.{self.task.dag_id}.{self.task.task_id}.duration', tags=self.stats_tags):
            self.task.params = context['params']
            with set_current_context(context):
                task_orig = self.render_templates(context=context)
            if not test_mode:
                rtif = RenderedTaskInstanceFields(ti=self, render_templates=False)
                RenderedTaskInstanceFields.write(rtif)
                RenderedTaskInstanceFields.delete_old_records(self.task_id, self.dag_id)
            airflow_context_vars = context_to_airflow_vars(context, in_env_var_format=True)
            os.environ.update(airflow_context_vars)
            if not self.next_method:
                self.log.info('Exporting env vars: %s', ' '.join((f'{k}={v!r}' for (k, v) in airflow_context_vars.items())))
            self.task.pre_execute(context=context)
            self._run_execute_callback(context, self.task)
            get_listener_manager().hook.on_task_instance_running(previous_state=TaskInstanceState.QUEUED, task_instance=self, session=session)
            with set_current_context(context):
                result = self._execute_task(context, task_orig)
            self.task.post_execute(context=context, result=result)
        Stats.incr(f'operator_successes_{self.task.task_type}', tags=self.stats_tags)
        Stats.incr('operator_successes', tags={**self.stats_tags, 'task_type': self.task.task_type})
        Stats.incr('ti_successes', tags=self.stats_tags)

    def _execute_task(self, context, task_orig):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute Task (optionally with a Timeout) and push Xcom results.\n\n        :param context: Jinja2 context\n        :param task_orig: origin task\n        '
        return _execute_task(self, context, task_orig)

    @provide_session
    def _defer_task(self, session: Session, defer: TaskDeferred) -> None:
        if False:
            while True:
                i = 10
        'Mark the task as deferred and sets up the trigger that is needed to resume it.'
        from airflow.models.trigger import Trigger
        trigger_row = Trigger.from_object(defer.trigger)
        session.add(trigger_row)
        session.flush()
        self.state = TaskInstanceState.DEFERRED
        self.trigger_id = trigger_row.id
        self.next_method = defer.method_name
        self.next_kwargs = defer.kwargs or {}
        self._try_number -= 1
        if defer.timeout is not None:
            self.trigger_timeout = timezone.utcnow() + defer.timeout
        else:
            self.trigger_timeout = None
        execution_timeout = self.task.execution_timeout
        if execution_timeout:
            if self.trigger_timeout:
                self.trigger_timeout = min(self.start_date + execution_timeout, self.trigger_timeout)
            else:
                self.trigger_timeout = self.start_date + execution_timeout

    def _run_execute_callback(self, context: Context, task: Operator) -> None:
        if False:
            return 10
        'Functions that need to be run before a Task is executed.'
        callbacks = task.on_execute_callback
        if callbacks:
            callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
            for callback in callbacks:
                try:
                    callback(context)
                except Exception:
                    self.log.exception('Failed when executing execute callback')

    @provide_session
    def run(self, verbose: bool=True, ignore_all_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_task_deps: bool=False, ignore_ti_state: bool=False, mark_success: bool=False, test_mode: bool=False, job_id: str | None=None, pool: str | None=None, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        'Run TaskInstance.'
        res = self.check_and_change_state_before_execution(verbose=verbose, ignore_all_deps=ignore_all_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_task_deps=ignore_task_deps, ignore_ti_state=ignore_ti_state, mark_success=mark_success, test_mode=test_mode, job_id=job_id, pool=pool, session=session)
        if not res:
            return
        self._run_raw_task(mark_success=mark_success, test_mode=test_mode, job_id=job_id, pool=pool, session=session)

    def dry_run(self) -> None:
        if False:
            while True:
                i = 10
        'Only Renders Templates for the TI.'
        self.task = self.task.prepare_for_execution()
        self.render_templates()
        if TYPE_CHECKING:
            assert isinstance(self.task, BaseOperator)
        self.task.dry_run()

    @provide_session
    def _handle_reschedule(self, actual_start_date, reschedule_exception, test_mode=False, session=NEW_SESSION):
        if False:
            print('Hello World!')
        if test_mode:
            return
        from airflow.models.dagrun import DagRun
        self.refresh_from_db(session)
        self.end_date = timezone.utcnow()
        self.set_duration()
        with_row_locks(session.query(DagRun).filter_by(dag_id=self.dag_id, run_id=self.run_id), session=session).one()
        session.add(TaskReschedule(self.task, self.run_id, self._try_number, actual_start_date, self.end_date, reschedule_exception.reschedule_date, self.map_index))
        self.state = TaskInstanceState.UP_FOR_RESCHEDULE
        self._try_number -= 1
        self.clear_next_method_args()
        session.merge(self)
        session.commit()
        self.log.info('Rescheduling task, marking task as UP_FOR_RESCHEDULE')

    @staticmethod
    def get_truncated_error_traceback(error: BaseException, truncate_to: Callable) -> TracebackType | None:
        if False:
            print('Hello World!')
        '\n        Truncate the traceback of an exception to the first frame called from within a given function.\n\n        :param error: exception to get traceback from\n        :param truncate_to: Function to truncate TB to. Must have a ``__code__`` attribute\n\n        :meta private:\n        '
        tb = error.__traceback__
        code = truncate_to.__func__.__code__
        while tb is not None:
            if tb.tb_frame.f_code is code:
                return tb.tb_next
            tb = tb.tb_next
        return tb or error.__traceback__

    @classmethod
    @internal_api_call
    @provide_session
    def fetch_handle_failure_context(cls, ti: TaskInstance | TaskInstancePydantic, error: None | str | Exception | KeyboardInterrupt, test_mode: bool | None=None, context: Context | None=None, force_fail: bool=False, session: Session=NEW_SESSION):
        if False:
            for i in range(10):
                print('nop')
        'Handle Failure for the TaskInstance.'
        get_listener_manager().hook.on_task_instance_failed(previous_state=TaskInstanceState.RUNNING, task_instance=ti, session=session)
        if error:
            if isinstance(error, BaseException):
                tb = TaskInstance.get_truncated_error_traceback(error, truncate_to=ti._execute_task)
                cls.logger().error('Task failed with exception', exc_info=(type(error), error, tb))
            else:
                cls.logger().error('%s', error)
        if not test_mode:
            ti.refresh_from_db(session)
        ti.end_date = timezone.utcnow()
        ti.set_duration()
        Stats.incr(f'operator_failures_{ti.operator}', tags=ti.stats_tags)
        Stats.incr('operator_failures', tags={**ti.stats_tags, 'operator': ti.operator})
        Stats.incr('ti_failures', tags=ti.stats_tags)
        if not test_mode:
            session.add(Log(TaskInstanceState.FAILED.value, ti))
            session.add(TaskFail(ti=ti))
        ti.clear_next_method_args()
        if context is None and getattr(ti, 'task', None):
            context = ti.get_template_context(session)
        if context is not None:
            context['exception'] = error
        task: BaseOperator | None = None
        try:
            if getattr(ti, 'task', None) and context:
                task = ti.task.unmap((context, session))
        except Exception:
            cls.logger().error('Unable to unmap task to determine if we need to send an alert email')
        if force_fail or not ti.is_eligible_to_retry():
            ti.state = TaskInstanceState.FAILED
            email_for_state = operator.attrgetter('email_on_failure')
            callbacks = task.on_failure_callback if task else None
            if task and task.dag and task.dag.fail_stop:
                _stop_remaining_tasks(task_instance=ti, session=session)
        else:
            if ti.state == TaskInstanceState.QUEUED:
                ti._try_number += 1
            ti.state = State.UP_FOR_RETRY
            email_for_state = operator.attrgetter('email_on_retry')
            callbacks = task.on_retry_callback if task else None
        return {'ti': ti, 'email_for_state': email_for_state, 'task': task, 'callbacks': callbacks, 'context': context}

    @staticmethod
    @internal_api_call
    @provide_session
    def save_to_db(ti: TaskInstance | TaskInstancePydantic, session: Session=NEW_SESSION):
        if False:
            return 10
        session.merge(ti)
        session.flush()

    @provide_session
    def handle_failure(self, error: None | str | Exception | KeyboardInterrupt, test_mode: bool | None=None, context: Context | None=None, force_fail: bool=False, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Handle Failure for a task instance.\n\n        :param error: if specified, log the specific exception if thrown\n        :param session: SQLAlchemy ORM Session\n        :param test_mode: doesn't record success or failure in the DB if True\n        :param context: Jinja2 context\n        :param force_fail: if True, task does not retry\n        "
        _handle_failure(task_instance=self, error=error, session=session, test_mode=test_mode, context=context, force_fail=force_fail)

    def is_eligible_to_retry(self):
        if False:
            print('Hello World!')
        'Is task instance is eligible for retry.'
        return _is_eligible_to_retry(task_instance=self)

    def get_template_context(self, session: Session | None=None, ignore_param_exceptions: bool=True) -> Context:
        if False:
            return 10
        '\n        Return TI Context.\n\n        :param session: SQLAlchemy ORM Session\n        :param ignore_param_exceptions: flag to suppress value exceptions while initializing the ParamsDict\n        '
        return _get_template_context(task_instance=self, session=session, ignore_param_exceptions=ignore_param_exceptions)

    @provide_session
    def get_rendered_template_fields(self, session: Session=NEW_SESSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update task with rendered template fields for presentation in UI.\n\n        If task has already run, will fetch from DB; otherwise will render.\n        '
        from airflow.models.renderedtifields import RenderedTaskInstanceFields
        rendered_task_instance_fields = RenderedTaskInstanceFields.get_templated_fields(self, session=session)
        if rendered_task_instance_fields:
            self.task = self.task.unmap(None)
            for (field_name, rendered_value) in rendered_task_instance_fields.items():
                setattr(self.task, field_name, rendered_value)
            return
        try:
            from airflow.utils.log.secrets_masker import redact
            self.render_templates()
            for field_name in self.task.template_fields:
                rendered_value = getattr(self.task, field_name)
                setattr(self.task, field_name, redact(rendered_value, field_name))
        except (TemplateAssertionError, UndefinedError) as e:
            raise AirflowException("Webserver does not have access to User-defined Macros or Filters when Dag Serialization is enabled. Hence for the task that have not yet started running, please use 'airflow tasks render' for debugging the rendering of template_fields.") from e

    def overwrite_params_with_dag_run_conf(self, params, dag_run):
        if False:
            while True:
                i = 10
        'Overwrite Task Params with DagRun.conf.'
        if dag_run and dag_run.conf:
            self.log.debug('Updating task params (%s) with DagRun.conf (%s)', params, dag_run.conf)
            params.update(dag_run.conf)

    def render_templates(self, context: Context | None=None) -> Operator:
        if False:
            i = 10
            return i + 15
        'Render templates in the operator fields.\n\n        If the task was originally mapped, this may replace ``self.task`` with\n        the unmapped, fully rendered BaseOperator. The original ``self.task``\n        before replacement is returned.\n        '
        if not context:
            context = self.get_template_context()
        original_task = self.task
        original_task.render_template_fields(context)
        return original_task

    def render_k8s_pod_yaml(self) -> dict | None:
        if False:
            return 10
        'Render the k8s pod yaml.'
        try:
            from airflow.providers.cncf.kubernetes.template_rendering import render_k8s_pod_yaml as render_k8s_pod_yaml_from_provider
        except ImportError:
            raise RuntimeError('You need to have the `cncf.kubernetes` provider installed to use this feature. Also rather than calling it directly you should import render_k8s_pod_yaml from airflow.providers.cncf.kubernetes.template_rendering and call it with TaskInstance as the first argument.')
        warnings.warn('You should not call `task_instance.render_k8s_pod_yaml` directly. This method will be removedin Airflow 3. Rather than calling it directly you should import `render_k8s_pod_yaml` from `airflow.providers.cncf.kubernetes.template_rendering` and call it with `TaskInstance` as the first argument.', DeprecationWarning, stacklevel=2)
        return render_k8s_pod_yaml_from_provider(self)

    @provide_session
    def get_rendered_k8s_spec(self, session: Session=NEW_SESSION):
        if False:
            for i in range(10):
                print('nop')
        'Render the k8s pod yaml.'
        try:
            from airflow.providers.cncf.kubernetes.template_rendering import get_rendered_k8s_spec as get_rendered_k8s_spec_from_provider
        except ImportError:
            raise RuntimeError('You need to have the `cncf.kubernetes` provider installed to use this feature. Also rather than calling it directly you should import `get_rendered_k8s_spec` from `airflow.providers.cncf.kubernetes.template_rendering` and call it with `TaskInstance` as the first argument.')
        warnings.warn('You should not call `task_instance.render_k8s_pod_yaml` directly. This method will be removedin Airflow 3. Rather than calling it directly you should import `get_rendered_k8s_spec` from `airflow.providers.cncf.kubernetes.template_rendering` and call it with `TaskInstance` as the first argument.', DeprecationWarning, stacklevel=2)
        return get_rendered_k8s_spec_from_provider(self, session=session)

    def get_email_subject_content(self, exception: BaseException, task: BaseOperator | None=None) -> tuple[str, str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Get the email subject content for exceptions.\n\n        :param exception: the exception sent in the email\n        :param task:\n        '
        return _get_email_subject_content(task_instance=self, exception=exception, task=task)

    def email_alert(self, exception, task: BaseOperator) -> None:
        if False:
            while True:
                i = 10
        '\n        Send alert email with exception information.\n\n        :param exception: the exception\n        :param task: task related to the exception\n        '
        _email_alert(task_instance=self, exception=exception, task=task)

    def set_duration(self) -> None:
        if False:
            i = 10
            return i + 15
        'Set task instance duration.'
        _set_duration(task_instance=self)

    @provide_session
    def xcom_push(self, key: str, value: Any, execution_date: datetime | None=None, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Make an XCom available for tasks to pull.\n\n        :param key: Key to store the value under.\n        :param value: Value to store. What types are possible depends on whether\n            ``enable_xcom_pickling`` is true or not. If so, this can be any\n            picklable object; only be JSON-serializable may be used otherwise.\n        :param execution_date: Deprecated parameter that has no effect.\n        '
        if execution_date is not None:
            self_execution_date = self.get_dagrun(session).execution_date
            if execution_date < self_execution_date:
                raise ValueError(f'execution_date can not be in the past (current execution_date is {self_execution_date}; received {execution_date})')
            elif execution_date is not None:
                message = "Passing 'execution_date' to 'TaskInstance.xcom_push()' is deprecated."
                warnings.warn(message, RemovedInAirflow3Warning, stacklevel=3)
        XCom.set(key=key, value=value, task_id=self.task_id, dag_id=self.dag_id, run_id=self.run_id, map_index=self.map_index, session=session)

    @provide_session
    def xcom_pull(self, task_ids: str | Iterable[str] | None=None, dag_id: str | None=None, key: str=XCOM_RETURN_KEY, include_prior_dates: bool=False, session: Session=NEW_SESSION, *, map_indexes: int | Iterable[int] | None=None, default: Any=None) -> Any:
        if False:
            return 10
        "Pull XComs that optionally meet certain criteria.\n\n        :param key: A key for the XCom. If provided, only XComs with matching\n            keys will be returned. The default key is ``'return_value'``, also\n            available as constant ``XCOM_RETURN_KEY``. This key is automatically\n            given to XComs returned by tasks (as opposed to being pushed\n            manually). To remove the filter, pass *None*.\n        :param task_ids: Only XComs from tasks with matching ids will be\n            pulled. Pass *None* to remove the filter.\n        :param dag_id: If provided, only pulls XComs from this DAG. If *None*\n            (default), the DAG of the calling task is used.\n        :param map_indexes: If provided, only pull XComs with matching indexes.\n            If *None* (default), this is inferred from the task(s) being pulled\n            (see below for details).\n        :param include_prior_dates: If False, only XComs from the current\n            execution_date are returned. If *True*, XComs from previous dates\n            are returned as well.\n\n        When pulling one single task (``task_id`` is *None* or a str) without\n        specifying ``map_indexes``, the return value is inferred from whether\n        the specified task is mapped. If not, value from the one single task\n        instance is returned. If the task to pull is mapped, an iterator (not a\n        list) yielding XComs from mapped task instances is returned. In either\n        case, ``default`` (*None* if not specified) is returned if no matching\n        XComs are found.\n\n        When pulling multiple tasks (i.e. either ``task_id`` or ``map_index`` is\n        a non-str iterable), a list of matching XComs is returned. Elements in\n        the list is ordered by item ordering in ``task_id`` and ``map_index``.\n        "
        if dag_id is None:
            dag_id = self.dag_id
        query = XCom.get_many(key=key, run_id=self.run_id, dag_ids=dag_id, task_ids=task_ids, map_indexes=map_indexes, include_prior_dates=include_prior_dates, session=session)
        if (task_ids is None or isinstance(task_ids, str)) and (not isinstance(map_indexes, Iterable)):
            first = query.with_entities(XCom.run_id, XCom.task_id, XCom.dag_id, XCom.map_index, XCom.value).first()
            if first is None:
                return default
            if map_indexes is not None or first.map_index < 0:
                return XCom.deserialize_value(first)
            query = query.order_by(None).order_by(XCom.map_index.asc())
            return LazyXComAccess.build_from_xcom_query(query)
        query = query.order_by(None)
        if task_ids is None or isinstance(task_ids, str):
            query = query.order_by(XCom.task_id)
        else:
            task_id_whens = {tid: i for (i, tid) in enumerate(task_ids)}
            if task_id_whens:
                query = query.order_by(case(task_id_whens, value=XCom.task_id))
            else:
                query = query.order_by(XCom.task_id)
        if map_indexes is None or isinstance(map_indexes, int):
            query = query.order_by(XCom.map_index)
        elif isinstance(map_indexes, range):
            order = XCom.map_index
            if map_indexes.step < 0:
                order = order.desc()
            query = query.order_by(order)
        else:
            map_index_whens = {map_index: i for (i, map_index) in enumerate(map_indexes)}
            if map_index_whens:
                query = query.order_by(case(map_index_whens, value=XCom.map_index))
            else:
                query = query.order_by(XCom.map_index)
        return LazyXComAccess.build_from_xcom_query(query)

    @provide_session
    def get_num_running_task_instances(self, session: Session, same_dagrun=False) -> int:
        if False:
            while True:
                i = 10
        'Return Number of running TIs from the DB.'
        num_running_task_instances_query = session.query(func.count()).filter(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id == self.task_id, TaskInstance.state == TaskInstanceState.RUNNING)
        if same_dagrun:
            num_running_task_instances_query = num_running_task_instances_query.filter(TaskInstance.run_id == self.run_id)
        return num_running_task_instances_query.scalar()

    def init_run_context(self, raw: bool=False) -> None:
        if False:
            return 10
        'Set the log context.'
        self.raw = raw
        self._set_context(self)

    @staticmethod
    def filter_for_tis(tis: Iterable[TaskInstance | TaskInstanceKey]) -> BooleanClauseList | None:
        if False:
            while True:
                i = 10
        'Return SQLAlchemy filter to query selected task instances.'
        tis = list(tis)
        if not tis:
            return None
        first = tis[0]
        dag_id = first.dag_id
        run_id = first.run_id
        map_index = first.map_index
        first_task_id = first.task_id
        (dag_ids, run_ids, map_indices, task_ids) = (set(), set(), set(), set())
        for t in tis:
            dag_ids.add(t.dag_id)
            run_ids.add(t.run_id)
            map_indices.add(t.map_index)
            task_ids.add(t.task_id)
        if dag_ids == {dag_id} and run_ids == {run_id} and (map_indices == {map_index}):
            return and_(TaskInstance.dag_id == dag_id, TaskInstance.run_id == run_id, TaskInstance.map_index == map_index, TaskInstance.task_id.in_(task_ids))
        if dag_ids == {dag_id} and task_ids == {first_task_id} and (map_indices == {map_index}):
            return and_(TaskInstance.dag_id == dag_id, TaskInstance.run_id.in_(run_ids), TaskInstance.map_index == map_index, TaskInstance.task_id == first_task_id)
        if dag_ids == {dag_id} and run_ids == {run_id} and (task_ids == {first_task_id}):
            return and_(TaskInstance.dag_id == dag_id, TaskInstance.run_id == run_id, TaskInstance.map_index.in_(map_indices), TaskInstance.task_id == first_task_id)
        filter_condition = []
        task_id_groups: dict[tuple, dict[Any, list[Any]]] = defaultdict(lambda : defaultdict(list))
        map_index_groups: dict[tuple, dict[Any, list[Any]]] = defaultdict(lambda : defaultdict(list))
        for t in tis:
            task_id_groups[t.dag_id, t.run_id][t.task_id].append(t.map_index)
            map_index_groups[t.dag_id, t.run_id][t.map_index].append(t.task_id)
        for (cur_dag_id, cur_run_id) in itertools.product(dag_ids, run_ids):
            dag_task_id_groups = task_id_groups[cur_dag_id, cur_run_id]
            dag_map_index_groups = map_index_groups[cur_dag_id, cur_run_id]
            if len(dag_task_id_groups) <= len(dag_map_index_groups):
                for (cur_task_id, cur_map_indices) in dag_task_id_groups.items():
                    filter_condition.append(and_(TaskInstance.dag_id == cur_dag_id, TaskInstance.run_id == cur_run_id, TaskInstance.task_id == cur_task_id, TaskInstance.map_index.in_(cur_map_indices)))
            else:
                for (cur_map_index, cur_task_ids) in dag_map_index_groups.items():
                    filter_condition.append(and_(TaskInstance.dag_id == cur_dag_id, TaskInstance.run_id == cur_run_id, TaskInstance.task_id.in_(cur_task_ids), TaskInstance.map_index == cur_map_index))
        return or_(*filter_condition)

    @classmethod
    def ti_selector_condition(cls, vals: Collection[str | tuple[str, int]]) -> ColumnOperators:
        if False:
            i = 10
            return i + 15
        '\n        Build an SQLAlchemy filter for a list of task_ids or tuples of (task_id,map_index).\n\n        :meta private:\n        '
        task_id_only = [v for v in vals if isinstance(v, str)]
        with_map_index = [v for v in vals if not isinstance(v, str)]
        filters: list[ColumnOperators] = []
        if task_id_only:
            filters.append(cls.task_id.in_(task_id_only))
        if with_map_index:
            filters.append(tuple_in_condition((cls.task_id, cls.map_index), with_map_index))
        if not filters:
            return false()
        if len(filters) == 1:
            return filters[0]
        return or_(*filters)

    @classmethod
    @internal_api_call
    @Sentry.enrich_errors
    @provide_session
    def _schedule_downstream_tasks(cls, ti: TaskInstance | TaskInstancePydantic, session: Session=NEW_SESSION, max_tis_per_query: int | None=None):
        if False:
            return 10
        from sqlalchemy.exc import OperationalError
        from airflow.models.dagrun import DagRun
        try:
            dag_run = with_row_locks(session.query(DagRun).filter_by(dag_id=ti.dag_id, run_id=ti.run_id), session=session).one()
            task = ti.task
            if TYPE_CHECKING:
                assert task.dag
            partial_dag = task.dag.partial_subset(task.downstream_task_ids, include_downstream=True, include_upstream=False, include_direct_upstream=True)
            dag_run.dag = partial_dag
            info = dag_run.task_instance_scheduling_decisions(session)
            skippable_task_ids = {task_id for task_id in partial_dag.task_ids if task_id not in task.downstream_task_ids}
            schedulable_tis = [ti for ti in info.schedulable_tis if ti.task_id not in skippable_task_ids and (not (ti.task.inherits_from_empty_operator and (not ti.task.on_execute_callback) and (not ti.task.on_success_callback) and (not ti.task.outlets)))]
            for schedulable_ti in schedulable_tis:
                if not hasattr(schedulable_ti, 'task'):
                    schedulable_ti.task = task.dag.get_task(schedulable_ti.task_id)
            num = dag_run.schedule_tis(schedulable_tis, session=session, max_tis_per_query=max_tis_per_query)
            cls.logger().info('%d downstream tasks scheduled from follow-on schedule check', num)
            session.flush()
        except OperationalError as e:
            cls.logger().info('Skipping mini scheduling run due to exception: %s', e.statement, exc_info=True)
            session.rollback()

    @provide_session
    def schedule_downstream_tasks(self, session: Session=NEW_SESSION, max_tis_per_query: int | None=None):
        if False:
            return 10
        '\n        Schedule downstream tasks of this task instance.\n\n        :meta: private\n        '
        return TaskInstance._schedule_downstream_tasks(ti=self, session=session, max_tis_per_query=max_tis_per_query)

    def get_relevant_upstream_map_indexes(self, upstream: Operator, ti_count: int | None, *, session: Session) -> int | range | None:
        if False:
            print('Hello World!')
        'Infer the map indexes of an upstream "relevant" to this ti.\n\n        The bulk of the logic mainly exists to solve the problem described by\n        the following example, where \'val\' must resolve to different values,\n        depending on where the reference is being used::\n\n            @task\n            def this_task(v):  # This is self.task.\n                return v * 2\n\n            @task_group\n            def tg1(inp):\n                val = upstream(inp)  # This is the upstream task.\n                this_task(val)  # When inp is 1, val here should resolve to 2.\n                return val\n\n            # This val is the same object returned by tg1.\n            val = tg1.expand(inp=[1, 2, 3])\n\n            @task_group\n            def tg2(inp):\n                another_task(inp, val)  # val here should resolve to [2, 4, 6].\n\n            tg2.expand(inp=["a", "b"])\n\n        The surrounding mapped task groups of ``upstream`` and ``self.task`` are\n        inspected to find a common "ancestor". If such an ancestor is found,\n        we need to return specific map indexes to pull a partial value from\n        upstream XCom.\n\n        :param upstream: The referenced upstream task.\n        :param ti_count: The total count of task instance this task was expanded\n            by the scheduler, i.e. ``expanded_ti_count`` in the template context.\n        :return: Specific map index or map indexes to pull, or ``None`` if we\n            want to "whole" return value (i.e. no mapped task groups involved).\n        '
        if not ti_count:
            return None
        common_ancestor = _find_common_ancestor_mapped_group(self.task, upstream)
        if common_ancestor is None:
            return None
        ancestor_ti_count = common_ancestor.get_mapped_ti_count(self.run_id, session=session)
        ancestor_map_index = self.map_index * ancestor_ti_count // ti_count
        if not _is_further_mapped_inside(upstream, common_ancestor):
            return ancestor_map_index
        further_count = ti_count // ancestor_ti_count
        map_index_start = ancestor_map_index * further_count
        return range(map_index_start, map_index_start + further_count)

    def clear_db_references(self, session):
        if False:
            return 10
        '\n        Clear db tables that have a reference to this instance.\n\n        :param session: ORM Session\n\n        :meta private:\n        '
        from airflow.models.renderedtifields import RenderedTaskInstanceFields
        tables = [TaskFail, TaskInstanceNote, TaskReschedule, XCom, RenderedTaskInstanceFields]
        for table in tables:
            session.execute(delete(table).where(table.dag_id == self.dag_id, table.task_id == self.task_id, table.run_id == self.run_id, table.map_index == self.map_index))

def _find_common_ancestor_mapped_group(node1: Operator, node2: Operator) -> MappedTaskGroup | None:
    if False:
        return 10
    'Given two operators, find their innermost common mapped task group.'
    if node1.dag is None or node2.dag is None or node1.dag_id != node2.dag_id:
        return None
    parent_group_ids = {g.group_id for g in node1.iter_mapped_task_groups()}
    common_groups = (g for g in node2.iter_mapped_task_groups() if g.group_id in parent_group_ids)
    return next(common_groups, None)

def _is_further_mapped_inside(operator: Operator, container: TaskGroup) -> bool:
    if False:
        while True:
            i = 10
    'Whether given operator is *further* mapped inside a task group.'
    if isinstance(operator, MappedOperator):
        return True
    task_group = operator.task_group
    while task_group is not None and task_group.group_id != container.group_id:
        if isinstance(task_group, MappedTaskGroup):
            return True
        task_group = task_group.parent_group
    return False
TaskInstanceStateType = Tuple[TaskInstanceKey, TaskInstanceState]

class SimpleTaskInstance:
    """
    Simplified Task Instance.

    Used to send data between processes via Queues.
    """

    def __init__(self, dag_id: str, task_id: str, run_id: str, start_date: datetime | None, end_date: datetime | None, try_number: int, map_index: int, state: str, executor_config: Any, pool: str, queue: str, key: TaskInstanceKey, run_as_user: str | None=None, priority_weight: int | None=None):
        if False:
            print('Hello World!')
        self.dag_id = dag_id
        self.task_id = task_id
        self.run_id = run_id
        self.map_index = map_index
        self.start_date = start_date
        self.end_date = end_date
        self.try_number = try_number
        self.state = state
        self.executor_config = executor_config
        self.run_as_user = run_as_user
        self.pool = pool
        self.priority_weight = priority_weight
        self.queue = queue
        self.key = key

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('This method is deprecated. Use BaseSerialization.serialize.', RemovedInAirflow3Warning, stacklevel=2)
        new_dict = dict(self.__dict__)
        for key in new_dict:
            if key in ['start_date', 'end_date']:
                val = new_dict[key]
                if not val or isinstance(val, str):
                    continue
                new_dict.update({key: val.isoformat()})
        return new_dict

    @classmethod
    def from_ti(cls, ti: TaskInstance) -> SimpleTaskInstance:
        if False:
            print('Hello World!')
        return cls(dag_id=ti.dag_id, task_id=ti.task_id, run_id=ti.run_id, map_index=ti.map_index, start_date=ti.start_date, end_date=ti.end_date, try_number=ti.try_number, state=ti.state, executor_config=ti.executor_config, pool=ti.pool, queue=ti.queue, key=ti.key, run_as_user=ti.run_as_user if hasattr(ti, 'run_as_user') else None, priority_weight=ti.priority_weight if hasattr(ti, 'priority_weight') else None)

    @classmethod
    def from_dict(cls, obj_dict: dict) -> SimpleTaskInstance:
        if False:
            print('Hello World!')
        warnings.warn('This method is deprecated. Use BaseSerialization.deserialize.', RemovedInAirflow3Warning, stacklevel=2)
        ti_key = TaskInstanceKey(*obj_dict.pop('key'))
        start_date = None
        end_date = None
        start_date_str: str | None = obj_dict.pop('start_date')
        end_date_str: str | None = obj_dict.pop('end_date')
        if start_date_str:
            start_date = timezone.parse(start_date_str)
        if end_date_str:
            end_date = timezone.parse(end_date_str)
        return cls(**obj_dict, start_date=start_date, end_date=end_date, key=ti_key)

class TaskInstanceNote(Base):
    """For storage of arbitrary notes concerning the task instance."""
    __tablename__ = 'task_instance_note'
    user_id = Column(Integer, ForeignKey('ab_user.id', name='task_instance_note_user_fkey'), nullable=True)
    task_id = Column(StringID(), primary_key=True, nullable=False)
    dag_id = Column(StringID(), primary_key=True, nullable=False)
    run_id = Column(StringID(), primary_key=True, nullable=False)
    map_index = Column(Integer, primary_key=True, nullable=False)
    content = Column(String(1000).with_variant(Text(1000), 'mysql'))
    created_at = Column(UtcDateTime, default=timezone.utcnow, nullable=False)
    updated_at = Column(UtcDateTime, default=timezone.utcnow, onupdate=timezone.utcnow, nullable=False)
    task_instance = relationship('TaskInstance', back_populates='task_instance_note')
    __table_args__ = (PrimaryKeyConstraint('task_id', 'dag_id', 'run_id', 'map_index', name='task_instance_note_pkey', mssql_clustered=True), ForeignKeyConstraint((dag_id, task_id, run_id, map_index), ['task_instance.dag_id', 'task_instance.task_id', 'task_instance.run_id', 'task_instance.map_index'], name='task_instance_note_ti_fkey', ondelete='CASCADE'))

    def __init__(self, content, user_id=None):
        if False:
            for i in range(10):
                print('nop')
        self.content = content
        self.user_id = user_id

    def __repr__(self):
        if False:
            return 10
        prefix = f'<{self.__class__.__name__}: {self.dag_id}.{self.task_id} {self.run_id}'
        if self.map_index != -1:
            prefix += f' map_index={self.map_index}'
        return prefix + '>'
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from airflow.jobs.job import Job
    TaskInstance.queued_by_job = relationship(Job)