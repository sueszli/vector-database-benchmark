from __future__ import annotations
import contextlib
import datetime
import logging
import os
from collections import deque
from datetime import timedelta
from typing import Generator
from unittest import mock
from unittest.mock import MagicMock, patch
import psutil
import pytest
import time_machine
from sqlalchemy import func
import airflow.example_dags
from airflow import settings
from airflow.callbacks.callback_requests import DagCallbackRequest, SlaCallbackRequest, TaskCallbackRequest
from airflow.callbacks.database_callback_sink import DatabaseCallbackSink
from airflow.callbacks.pipe_callback_sink import PipeCallbackSink
from airflow.dag_processing.manager import DagFileProcessorAgent
from airflow.datasets import Dataset
from airflow.exceptions import AirflowException
from airflow.executors.base_executor import BaseExecutor
from airflow.executors.executor_constants import MOCK_EXECUTOR
from airflow.executors.executor_loader import ExecutorLoader
from airflow.jobs.backfill_job_runner import BackfillJobRunner
from airflow.jobs.job import Job, run_job
from airflow.jobs.local_task_job_runner import LocalTaskJobRunner
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.models.dag import DAG, DagModel
from airflow.models.dagbag import DagBag
from airflow.models.dagrun import DagRun
from airflow.models.dataset import DatasetDagRunQueue, DatasetEvent, DatasetModel
from airflow.models.db_callback_request import DbCallbackRequest
from airflow.models.pool import Pool
from airflow.models.serialized_dag import SerializedDagModel
from airflow.models.taskinstance import SimpleTaskInstance, TaskInstance, TaskInstanceKey
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.serialization.serialized_objects import SerializedDAG
from airflow.utils import timezone
from airflow.utils.file import list_py_file_paths
from airflow.utils.session import create_session, provide_session
from airflow.utils.state import DagRunState, JobState, State, TaskInstanceState
from airflow.utils.types import DagRunType
from tests.listeners import dag_listener
from tests.listeners.test_listeners import get_listener_manager
from tests.models import TEST_DAGS_FOLDER
from tests.test_utils.asserts import assert_queries_count
from tests.test_utils.config import conf_vars, env_vars
from tests.test_utils.db import clear_db_dags, clear_db_datasets, clear_db_import_errors, clear_db_jobs, clear_db_pools, clear_db_runs, clear_db_serialized_dags, clear_db_sla_miss, set_default_pool_slots
from tests.test_utils.mock_executor import MockExecutor
from tests.test_utils.mock_operators import CustomOperator
from tests.utils.test_timezone import UTC
pytestmark = pytest.mark.db_test
ROOT_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
PERF_DAGS_FOLDER = os.path.join(ROOT_FOLDER, 'tests', 'test_utils', 'perf', 'dags')
ELASTIC_DAG_FILE = os.path.join(PERF_DAGS_FOLDER, 'elastic_dag.py')
TEST_DAG_FOLDER = os.environ['AIRFLOW__CORE__DAGS_FOLDER']
DEFAULT_DATE = timezone.datetime(2016, 1, 1)
TRY_NUMBER = 1

@pytest.fixture(scope='class')
def disable_load_example():
    if False:
        print('Hello World!')
    with conf_vars({('core', 'load_examples'): 'false'}):
        with env_vars({'AIRFLOW__CORE__LOAD_EXAMPLES': 'false'}):
            yield

@pytest.fixture(scope='module')
def dagbag():
    if False:
        i = 10
        return i + 15
    non_serialized_dagbag = DagBag(read_dags_from_db=False, include_examples=False)
    non_serialized_dagbag.sync_to_db()
    return DagBag(read_dags_from_db=True)

@pytest.fixture
def load_examples():
    if False:
        print('Hello World!')
    with conf_vars({('core', 'load_examples'): 'True'}):
        yield

@patch.dict(ExecutorLoader.executors, {MOCK_EXECUTOR: f'{MockExecutor.__module__}.{MockExecutor.__qualname__}'})
@pytest.mark.usefixtures('disable_load_example')
@pytest.mark.need_serialized_dag
class TestSchedulerJob:

    @staticmethod
    def clean_db():
        if False:
            print('Hello World!')
        clear_db_runs()
        clear_db_pools()
        clear_db_dags()
        clear_db_sla_miss()
        clear_db_import_errors()
        clear_db_jobs()
        clear_db_datasets()

    @pytest.fixture(autouse=True)
    def per_test(self) -> Generator:
        if False:
            return 10
        self.clean_db()
        self.job_runner = None
        yield
        if self.job_runner and self.job_runner.processor_agent:
            self.job_runner.processor_agent.end()
            self.job_runner = None
        self.clean_db()

    @pytest.fixture(autouse=True)
    def set_instance_attrs(self, dagbag) -> Generator:
        if False:
            for i in range(10):
                print('nop')
        self.dagbag: DagBag = dagbag
        self.null_exec: MockExecutor | None = MockExecutor()
        with patch('airflow.dag_processing.manager.SerializedDagModel.remove_deleted_dags'), patch('airflow.models.dag.DagCode.bulk_sync_to_db'):
            yield
        self.null_exec = None
        del self.dagbag

    @pytest.mark.parametrize('configs', [{('scheduler', 'standalone_dag_processor'): 'False'}, {('scheduler', 'standalone_dag_processor'): 'True'}])
    def test_is_alive(self, configs):
        if False:
            print('Hello World!')
        with conf_vars(configs):
            scheduler_job = Job(heartrate=10, state=State.RUNNING)
            self.job_runner = SchedulerJobRunner(scheduler_job)
            assert scheduler_job.is_alive()
            scheduler_job.latest_heartbeat = timezone.utcnow() - datetime.timedelta(seconds=20)
            assert scheduler_job.is_alive()
            scheduler_job.latest_heartbeat = timezone.utcnow() - datetime.timedelta(seconds=31)
            assert not scheduler_job.is_alive()
            scheduler_job.latest_heartbeat = timezone.utcnow() - datetime.timedelta(days=1)
            assert not scheduler_job.is_alive()
            scheduler_job.state = State.SUCCESS
            scheduler_job.latest_heartbeat = timezone.utcnow() - datetime.timedelta(seconds=10)
            assert not scheduler_job.is_alive(), 'Completed jobs even with recent heartbeat should not be alive'

    def run_single_scheduler_loop_with_no_dags(self, dags_folder):
        if False:
            while True:
                i = 10
        '\n        Utility function that runs a single scheduler loop without actually\n        changing/scheduling any dags. This is useful to simulate the other side effects of\n        running a scheduler loop, e.g. to see what parse errors there are in the\n        dags_folder.\n\n        :param dags_folder: the directory to traverse\n        '
        scheduler_job = Job(executor=self.null_exec, num_times_parse_dags=1, subdir=os.path.join(dags_folder))
        self.job_runner = SchedulerJobRunner(scheduler_job)
        scheduler_job.heartrate = 0
        run_job(scheduler_job, execute_callable=self.job_runner._execute)

    def test_no_orphan_process_will_be_left(self, tmp_path):
        if False:
            return 10
        current_process = psutil.Process()
        old_children = current_process.children(recursive=True)
        scheduler_job = Job(executor=MockExecutor(do_update=False))
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.fspath(tmp_path), num_runs=1)
        run_job(scheduler_job, execute_callable=self.job_runner._execute)
        current_children = set(current_process.children(recursive=True)) - set(old_children)
        assert not current_children

    @mock.patch('airflow.jobs.scheduler_job_runner.TaskCallbackRequest')
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.incr')
    def test_process_executor_events(self, mock_stats_incr, mock_task_callback, dag_maker):
        if False:
            while True:
                i = 10
        dag_id = 'test_process_executor_events'
        task_id_1 = 'dummy_task'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, fileloc='/test_path1/'):
            task1 = EmptyOperator(task_id=task_id_1)
        ti1 = dag_maker.create_dagrun().get_task_instance(task1.task_id)
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        ti1.state = State.QUEUED
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.FAILED, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.FAILED
        scheduler_job.executor.callback_sink.send.assert_not_called()
        self.job_runner.processor_agent.reset_mock()
        ti1.state = State.SUCCESS
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.SUCCESS, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.SUCCESS
        scheduler_job.executor.callback_sink.send.assert_not_called()
        mock_stats_incr.assert_has_calls([mock.call('scheduler.tasks.killed_externally', tags={'dag_id': dag_id, 'task_id': ti1.task_id}), mock.call('operator_failures_EmptyOperator', tags={'dag_id': dag_id, 'task_id': ti1.task_id}), mock.call('ti_failures', tags={'dag_id': dag_id, 'task_id': ti1.task_id})], any_order=True)

    @mock.patch('airflow.jobs.scheduler_job_runner.TaskCallbackRequest')
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.incr')
    def test_process_executor_events_with_no_callback(self, mock_stats_incr, mock_task_callback, dag_maker):
        if False:
            i = 10
            return i + 15
        dag_id = 'test_process_executor_events_with_no_callback'
        task_id = 'test_task'
        run_id = 'test_run'
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        session = settings.Session()
        with dag_maker(dag_id=dag_id, fileloc='/test_path1/'):
            task1 = EmptyOperator(task_id=task_id, retries=1)
        ti1 = dag_maker.create_dagrun(run_id=run_id, execution_date=DEFAULT_DATE + timedelta(hours=1)).get_task_instance(task1.task_id)
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        ti1.state = State.QUEUED
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.FAILED, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.UP_FOR_RETRY
        scheduler_job.executor.callback_sink.send.assert_not_called()
        ti1.state = State.SUCCESS
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.SUCCESS, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.SUCCESS
        scheduler_job.executor.callback_sink.send.assert_not_called()
        mock_stats_incr.assert_has_calls([mock.call('scheduler.tasks.killed_externally', tags={'dag_id': dag_id, 'task_id': task_id}), mock.call('operator_failures_EmptyOperator', tags={'dag_id': dag_id, 'task_id': task_id}), mock.call('ti_failures', tags={'dag_id': dag_id, 'task_id': task_id})], any_order=True)

    @mock.patch('airflow.jobs.scheduler_job_runner.TaskCallbackRequest')
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.incr')
    def test_process_executor_events_with_callback(self, mock_stats_incr, mock_task_callback, dag_maker):
        if False:
            while True:
                i = 10
        dag_id = 'test_process_executor_events_with_callback'
        task_id_1 = 'dummy_task'
        with dag_maker(dag_id=dag_id, fileloc='/test_path1/') as dag:
            task1 = EmptyOperator(task_id=task_id_1, on_failure_callback=lambda x: print('hi'))
        ti1 = dag_maker.create_dagrun().get_task_instance(task1.task_id)
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        session = settings.Session()
        ti1.state = State.QUEUED
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.FAILED, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db()
        assert ti1.state == State.QUEUED
        mock_task_callback.assert_called_once_with(full_filepath=dag.fileloc, simple_task_instance=mock.ANY, processor_subdir=None, msg="Executor reports task instance <TaskInstance: test_process_executor_events_with_callback.dummy_task test [queued]> finished (failed) although the task says it's queued. (Info: None) Was the task killed externally?")
        scheduler_job.executor.callback_sink.send.assert_called_once_with(task_callback)
        scheduler_job.executor.callback_sink.reset_mock()
        mock_stats_incr.assert_called_once_with('scheduler.tasks.killed_externally', tags={'dag_id': 'test_process_executor_events_with_callback', 'task_id': 'dummy_task'})

    @mock.patch('airflow.jobs.scheduler_job_runner.TaskCallbackRequest')
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.incr')
    def test_process_executor_event_missing_dag(self, mock_stats_incr, mock_task_callback, dag_maker, caplog):
        if False:
            return 10
        dag_id = 'test_process_executor_events_with_callback'
        task_id_1 = 'dummy_task'
        with dag_maker(dag_id=dag_id, fileloc='/test_path1/'):
            task1 = EmptyOperator(task_id=task_id_1, on_failure_callback=lambda x: print('hi'))
        ti1 = dag_maker.create_dagrun().get_task_instance(task1.task_id)
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.dagbag = mock.MagicMock()
        self.job_runner.dagbag.get_dag.side_effect = Exception('failed')
        self.job_runner.processor_agent = mock.MagicMock()
        session = settings.Session()
        ti1.state = State.QUEUED
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.FAILED, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db()
        assert ti1.state == State.FAILED

    @mock.patch('airflow.jobs.scheduler_job_runner.TaskCallbackRequest')
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.incr')
    def test_process_executor_events_ti_requeued(self, mock_stats_incr, mock_task_callback, dag_maker):
        if False:
            print('Hello World!')
        dag_id = 'test_process_executor_events_ti_requeued'
        task_id_1 = 'dummy_task'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, fileloc='/test_path1/'):
            task1 = EmptyOperator(task_id=task_id_1)
        ti1 = dag_maker.create_dagrun().get_task_instance(task1.task_id)
        mock_stats_incr.reset_mock()
        executor = MockExecutor(do_update=False)
        task_callback = mock.MagicMock()
        mock_task_callback.return_value = task_callback
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.id = 1
        self.job_runner.processor_agent = mock.MagicMock()
        ti1.state = State.QUEUED
        ti1.queued_by_job_id = 1
        ti1.try_number = 2
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key.with_try_number(1)] = (State.SUCCESS, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.QUEUED
        scheduler_job.executor.callback_sink.send.assert_not_called()
        ti1.state = State.QUEUED
        ti1.queued_by_job_id = 2
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.SUCCESS, None)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.QUEUED
        scheduler_job.executor.callback_sink.send.assert_not_called()
        ti1.state = State.QUEUED
        ti1.queued_by_job_id = 1
        session.merge(ti1)
        session.commit()
        executor.event_buffer[ti1.key] = (State.SUCCESS, None)
        executor.has_task = mock.MagicMock(return_value=True)
        self.job_runner._process_executor_events(session=session)
        ti1.refresh_from_db(session=session)
        assert ti1.state == State.QUEUED
        scheduler_job.executor.callback_sink.send.assert_not_called()
        mock_stats_incr.assert_not_called()

    def test_execute_task_instances_is_paused_wont_execute(self, session, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_execute_task_instances_is_paused_wont_execute'
        task_id_1 = 'dummy_task'
        with dag_maker(dag_id=dag_id, session=session) as dag:
            EmptyOperator(task_id=task_id_1)
        assert isinstance(dag, SerializedDAG)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.BACKFILL_JOB)
        (ti1,) = dr1.task_instances
        ti1.state = State.SCHEDULED
        self.job_runner._critical_section_enqueue_task_instances(session)
        session.flush()
        ti1.refresh_from_db(session=session)
        assert State.SCHEDULED == ti1.state
        session.rollback()

    def test_execute_task_instances_backfill_tasks_wont_execute(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that backfill tasks won't get executed.\n        "
        dag_id = 'SchedulerJobTest.test_execute_task_instances_backfill_tasks_wont_execute'
        task_id_1 = 'dummy_task'
        with dag_maker(dag_id=dag_id):
            task1 = EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.BACKFILL_JOB)
        ti1 = TaskInstance(task1, run_id=dr1.run_id)
        ti1.refresh_from_db()
        ti1.state = State.SCHEDULED
        session.merge(ti1)
        session.flush()
        assert dr1.is_backfill
        self.job_runner._critical_section_enqueue_task_instances(session)
        session.flush()
        ti1.refresh_from_db()
        assert State.SCHEDULED == ti1.state
        session.rollback()

    @conf_vars({('scheduler', 'standalone_dag_processor'): 'False'})
    def test_setup_callback_sink_not_standalone_dag_processor(self):
        if False:
            print('Hello World!')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull, num_runs=1)
        self.job_runner._execute()
        assert isinstance(scheduler_job.executor.callback_sink, PipeCallbackSink)

    @conf_vars({('scheduler', 'standalone_dag_processor'): 'True'})
    def test_setup_callback_sink_standalone_dag_processor(self):
        if False:
            print('Hello World!')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull, num_runs=1)
        self.job_runner._execute()
        assert isinstance(scheduler_job.executor.callback_sink, DatabaseCallbackSink)

    def test_find_executable_task_instances_backfill(self, dag_maker):
        if False:
            print('Hello World!')
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_backfill'
        task_id_1 = 'dummy'
        with dag_maker(dag_id=dag_id, max_active_tasks=16):
            task1 = EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.BACKFILL_JOB, state=State.RUNNING)
        ti_backfill = dr2.get_task_instance(task1.task_id)
        ti_with_dagrun = dr1.get_task_instance(task1.task_id)
        ti_backfill.state = State.SCHEDULED
        ti_with_dagrun.state = State.SCHEDULED
        session.merge(dr2)
        session.merge(ti_backfill)
        session.merge(ti_with_dagrun)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        res_keys = (x.key for x in res)
        assert ti_with_dagrun.key in res_keys
        session.rollback()

    def test_find_executable_task_instances_pool(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_pool'
        task_id_1 = 'dummy'
        task_id_2 = 'dummydummy'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id_1, pool='a', priority_weight=2)
            EmptyOperator(task_id=task_id_2, pool='b', priority_weight=1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        tis = [dr1.get_task_instance(task_id_1, session=session), dr1.get_task_instance(task_id_2, session=session), dr2.get_task_instance(task_id_1, session=session), dr2.get_task_instance(task_id_2, session=session)]
        tis.sort(key=lambda ti: ti.key)
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        pool = Pool(pool='a', slots=1, description='haha', include_deferred=False)
        pool2 = Pool(pool='b', slots=100, description='haha', include_deferred=False)
        session.add(pool)
        session.add(pool2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        session.flush()
        assert 3 == len(res)
        res_keys = []
        for ti in res:
            res_keys.append(ti.key)
        assert tis[0].key in res_keys
        assert tis[2].key in res_keys
        assert tis[3].key in res_keys
        session.rollback()

    @pytest.mark.parametrize('state, total_executed_ti', [(DagRunState.SUCCESS, 0), (DagRunState.FAILED, 0), (DagRunState.RUNNING, 2), (DagRunState.QUEUED, 0)])
    def test_find_executable_task_instances_only_running_dagruns(self, state, total_executed_ti, dag_maker, session):
        if False:
            while True:
                i = 10
        "Test that only task instances of 'running' dagruns are executed"
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_only_running_dagruns'
        task_id_1 = 'dummy'
        task_id_2 = 'dummydummy'
        with dag_maker(dag_id=dag_id, session=session):
            EmptyOperator(task_id=task_id_1)
            EmptyOperator(task_id=task_id_2)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr = dag_maker.create_dagrun(state=state)
        tis = dr.task_instances
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        session.flush()
        assert total_executed_ti == len(res)

    def test_find_executable_task_instances_order_execution_date(self, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Test that task instances follow execution_date order priority. If two dagruns with\n        different execution dates are scheduled, tasks with earliest dagrun execution date will first\n        be executed\n        '
        dag_id_1 = 'SchedulerJobTest.test_find_executable_task_instances_order_execution_date-a'
        dag_id_2 = 'SchedulerJobTest.test_find_executable_task_instances_order_execution_date-b'
        task_id = 'task-a'
        session = settings.Session()
        with dag_maker(dag_id=dag_id_1, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id)
        dr1 = dag_maker.create_dagrun(execution_date=DEFAULT_DATE + timedelta(hours=1))
        with dag_maker(dag_id=dag_id_2, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id)
        dr2 = dag_maker.create_dagrun()
        dr1 = session.merge(dr1, load=False)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        tis = dr1.task_instances + dr2.task_instances
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        session.flush()
        assert [ti.key for ti in res] == [tis[1].key]
        session.rollback()

    def test_find_executable_task_instances_order_priority(self, dag_maker):
        if False:
            while True:
                i = 10
        dag_id_1 = 'SchedulerJobTest.test_find_executable_task_instances_order_priority-a'
        dag_id_2 = 'SchedulerJobTest.test_find_executable_task_instances_order_priority-b'
        task_id = 'task-a'
        session = settings.Session()
        with dag_maker(dag_id=dag_id_1, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id, priority_weight=1)
        dr1 = dag_maker.create_dagrun()
        with dag_maker(dag_id=dag_id_2, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id, priority_weight=4)
        dr2 = dag_maker.create_dagrun()
        dr1 = session.merge(dr1, load=False)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        tis = dr1.task_instances + dr2.task_instances
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        session.flush()
        assert [ti.key for ti in res] == [tis[1].key]
        session.rollback()

    def test_find_executable_task_instances_order_priority_with_pools(self, dag_maker):
        if False:
            return 10
        '\n        The scheduler job should pick tasks with higher priority for execution\n        even if different pools are involved.\n        '
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_order_priority_with_pools'
        session.add(Pool(pool='pool1', slots=32, include_deferred=False))
        session.add(Pool(pool='pool2', slots=32, include_deferred=False))
        with dag_maker(dag_id=dag_id, max_active_tasks=2):
            op1 = EmptyOperator(task_id='dummy1', priority_weight=1, pool='pool1')
            op2 = EmptyOperator(task_id='dummy2', priority_weight=2, pool='pool2')
            op3 = EmptyOperator(task_id='dummy3', priority_weight=3, pool='pool1')
        dag_run = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1 = dag_run.get_task_instance(op1.task_id, session)
        ti2 = dag_run.get_task_instance(op2.task_id, session)
        ti3 = dag_run.get_task_instance(op3.task_id, session)
        ti1.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        ti3.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 2 == len(res)
        assert ti3.key == res[0].key
        assert ti2.key == res[1].key
        session.rollback()

    def test_find_executable_task_instances_order_execution_date_and_priority(self, dag_maker):
        if False:
            print('Hello World!')
        dag_id_1 = 'SchedulerJobTest.test_find_executable_task_instances_order_execution_date_and_priority-a'
        dag_id_2 = 'SchedulerJobTest.test_find_executable_task_instances_order_execution_date_and_priority-b'
        task_id = 'task-a'
        session = settings.Session()
        with dag_maker(dag_id=dag_id_1, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id, priority_weight=1)
        dr1 = dag_maker.create_dagrun()
        with dag_maker(dag_id=dag_id_2, max_active_tasks=16, session=session):
            EmptyOperator(task_id=task_id, priority_weight=4)
        dr2 = dag_maker.create_dagrun(execution_date=DEFAULT_DATE + timedelta(hours=1))
        dr1 = session.merge(dr1, load=False)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        tis = dr1.task_instances + dr2.task_instances
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        session.flush()
        assert [ti.key for ti in res] == [tis[1].key]
        session.rollback()

    def test_find_executable_task_instances_in_default_pool(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        set_default_pool_slots(1)
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_in_default_pool'
        with dag_maker(dag_id=dag_id):
            op1 = EmptyOperator(task_id='dummy1')
            op2 = EmptyOperator(task_id='dummy2')
        scheduler_job = Job(executor=MockExecutor())
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull, num_runs=1)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED, state=State.RUNNING)
        ti1 = dr1.get_task_instance(op1.task_id, session)
        ti2 = dr2.get_task_instance(op2.task_id, session)
        ti1.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        ti2.state = State.RUNNING
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 0 == len(res)
        session.rollback()
        session.close()

    def test_queued_task_instances_fails_with_missing_dag(self, dag_maker, session):
        if False:
            i = 10
            return i + 15
        'Check that task instances of missing DAGs are failed'
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_not_in_dagbag'
        task_id_1 = 'dummy'
        task_id_2 = 'dummydummy'
        with dag_maker(dag_id=dag_id, session=session, default_args={'max_active_tis_per_dag': 1}):
            EmptyOperator(task_id=task_id_1)
            EmptyOperator(task_id=task_id_2)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.dagbag = mock.MagicMock()
        self.job_runner.dagbag.get_dag.return_value = None
        dr = dag_maker.create_dagrun(state=DagRunState.RUNNING)
        tis = dr.task_instances
        for ti in tis:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        session.flush()
        assert 0 == len(res)
        tis = dr.get_task_instances(session=session)
        assert len(tis) == 2
        assert all((ti.state == State.FAILED for ti in tis))

    def test_nonexistent_pool(self, dag_maker):
        if False:
            i = 10
            return i + 15
        dag_id = 'SchedulerJobTest.test_nonexistent_pool'
        with dag_maker(dag_id=dag_id, max_active_tasks=16):
            EmptyOperator(task_id='dummy_wrong_pool', pool='this_pool_doesnt_exist')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.task_instances[0]
        ti.state = State.SCHEDULED
        session.merge(ti)
        session.commit()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        session.flush()
        assert 0 == len(res)
        session.rollback()

    def test_infinite_pool(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_infinite_pool'
        with dag_maker(dag_id=dag_id, concurrency=16):
            EmptyOperator(task_id='dummy', pool='infinite_pool')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.task_instances[0]
        ti.state = State.SCHEDULED
        session.merge(ti)
        infinite_pool = Pool(pool='infinite_pool', slots=-1, description='infinite pool', include_deferred=False)
        session.add(infinite_pool)
        session.commit()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        session.flush()
        assert 1 == len(res)
        session.rollback()

    def test_not_enough_pool_slots(self, caplog, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_test_not_enough_pool_slots'
        with dag_maker(dag_id=dag_id, concurrency=16):
            EmptyOperator(task_id='cannot_run', pool='some_pool', pool_slots=4)
            EmptyOperator(task_id='can_run', pool='some_pool', pool_slots=1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.task_instances[0]
        ti.state = State.SCHEDULED
        session.merge(ti)
        ti = dr.task_instances[1]
        ti.state = State.SCHEDULED
        session.merge(ti)
        some_pool = Pool(pool='some_pool', slots=2, description='my pool', include_deferred=False)
        session.add(some_pool)
        session.commit()
        with caplog.at_level(logging.WARNING):
            self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
            assert "Not executing <TaskInstance: SchedulerJobTest.test_test_not_enough_pool_slots.cannot_run test [scheduled]>. Requested pool slots (4) are greater than total pool slots: '2' for pool: some_pool" in caplog.text
        assert session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id, TaskInstance.state == State.SCHEDULED).count() == 1
        assert session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id, TaskInstance.state == State.QUEUED).count() == 1
        session.flush()
        session.rollback()

    def test_find_executable_task_instances_none(self, dag_maker):
        if False:
            print('Hello World!')
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_none'
        task_id_1 = 'dummy'
        with dag_maker(dag_id=dag_id, max_active_tasks=16):
            EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        assert 0 == len(self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session))
        session.rollback()

    def test_tis_for_queued_dagruns_are_not_run(self, dag_maker):
        if False:
            i = 10
            return i + 15
        '\n        This tests that tis from queued dagruns are not queued\n        '
        dag_id = 'test_tis_for_queued_dagruns_are_not_run'
        task_id_1 = 'dummy'
        with dag_maker(dag_id):
            task1 = EmptyOperator(task_id=task_id_1)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        ti1 = TaskInstance(task1, run_id=dr1.run_id)
        ti2 = TaskInstance(task1, run_id=dr2.run_id)
        ti1.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        session.merge(ti1)
        session.merge(ti2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        assert ti2.key == res[0].key
        ti1.refresh_from_db()
        ti2.refresh_from_db()
        assert ti1.state == State.SCHEDULED
        assert ti2.state == State.QUEUED

    def test_find_executable_task_instances_concurrency(self, dag_maker):
        if False:
            return 10
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_concurrency'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, max_active_tasks=2, session=session):
            EmptyOperator(task_id='dummy')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        dr3 = dag_maker.create_dagrun_after(dr2, run_type=DagRunType.SCHEDULED)
        ti1 = dr1.task_instances[0]
        ti2 = dr2.task_instances[0]
        ti3 = dr3.task_instances[0]
        ti1.state = State.RUNNING
        ti2.state = State.SCHEDULED
        ti3.state = State.SCHEDULED
        session.merge(ti1)
        session.merge(ti2)
        session.merge(ti3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        res_keys = (x.key for x in res)
        assert ti2.key in res_keys
        ti2.state = State.RUNNING
        session.merge(ti2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 0 == len(res)
        session.rollback()

    def test_find_executable_task_instances_concurrency_queued(self, dag_maker):
        if False:
            return 10
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_concurrency_queued'
        with dag_maker(dag_id=dag_id, max_active_tasks=3):
            task1 = EmptyOperator(task_id='dummy1')
            task2 = EmptyOperator(task_id='dummy2')
            task3 = EmptyOperator(task_id='dummy3')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_run = dag_maker.create_dagrun()
        ti1 = dag_run.get_task_instance(task1.task_id)
        ti2 = dag_run.get_task_instance(task2.task_id)
        ti3 = dag_run.get_task_instance(task3.task_id)
        ti1.state = State.RUNNING
        ti2.state = State.QUEUED
        ti3.state = State.SCHEDULED
        session.merge(ti1)
        session.merge(ti2)
        session.merge(ti3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        assert res[0].key == ti3.key
        session.rollback()

    def test_find_executable_task_instances_max_active_tis_per_dag(self, dag_maker):
        if False:
            return 10
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_max_active_tis_per_dag'
        task_id_1 = 'dummy'
        task_id_2 = 'dummy2'
        with dag_maker(dag_id=dag_id, max_active_tasks=16):
            task1 = EmptyOperator(task_id=task_id_1, max_active_tis_per_dag=2)
            task2 = EmptyOperator(task_id=task_id_2)
        executor = MockExecutor(do_update=True)
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        dr3 = dag_maker.create_dagrun_after(dr2, run_type=DagRunType.SCHEDULED)
        ti1_1 = dr1.get_task_instance(task1.task_id)
        ti2 = dr1.get_task_instance(task2.task_id)
        ti1_1.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        session.merge(ti1_1)
        session.merge(ti2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 2 == len(res)
        ti1_1.state = State.RUNNING
        ti2.state = State.RUNNING
        ti1_2 = dr2.get_task_instance(task1.task_id)
        ti1_2.state = State.SCHEDULED
        session.merge(ti1_1)
        session.merge(ti2)
        session.merge(ti1_2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        ti1_2.state = State.RUNNING
        ti1_3 = dr3.get_task_instance(task1.task_id)
        ti1_3.state = State.SCHEDULED
        session.merge(ti1_2)
        session.merge(ti1_3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 0 == len(res)
        ti1_1.state = State.SCHEDULED
        ti1_2.state = State.SCHEDULED
        ti1_3.state = State.SCHEDULED
        session.merge(ti1_1)
        session.merge(ti1_2)
        session.merge(ti1_3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 2 == len(res)
        ti1_1.state = State.RUNNING
        ti1_2.state = State.SCHEDULED
        ti1_3.state = State.SCHEDULED
        session.merge(ti1_1)
        session.merge(ti1_2)
        session.merge(ti1_3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        session.rollback()

    def test_change_state_for_executable_task_instances_no_tis_with_state(self, dag_maker):
        if False:
            print('Hello World!')
        dag_id = 'SchedulerJobTest.test_change_state_for__no_tis_with_state'
        task_id_1 = 'dummy'
        with dag_maker(dag_id=dag_id, max_active_tasks=2):
            task1 = EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        dr3 = dag_maker.create_dagrun_after(dr2, run_type=DagRunType.SCHEDULED)
        ti1 = dr1.get_task_instance(task1.task_id)
        ti2 = dr2.get_task_instance(task1.task_id)
        ti3 = dr3.get_task_instance(task1.task_id)
        ti1.state = State.RUNNING
        ti2.state = State.RUNNING
        ti3.state = State.RUNNING
        session.merge(ti1)
        session.merge(ti2)
        session.merge(ti3)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=100, session=session)
        assert 0 == len(res)
        session.rollback()

    def test_find_executable_task_instances_not_enough_pool_slots_for_first(self, dag_maker):
        if False:
            while True:
                i = 10
        set_default_pool_slots(1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_not_enough_pool_slots_for_first'
        with dag_maker(dag_id=dag_id):
            op1 = EmptyOperator(task_id='dummy1', priority_weight=2, pool_slots=2)
            op2 = EmptyOperator(task_id='dummy2', priority_weight=1, pool_slots=1)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1 = dr1.get_task_instance(op1.task_id, session)
        ti2 = dr1.get_task_instance(op2.task_id, session)
        ti1.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        assert res[0].key == ti2.key
        session.rollback()

    def test_find_executable_task_instances_not_enough_dag_concurrency_for_first(self, dag_maker):
        if False:
            while True:
                i = 10
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id_1 = 'SchedulerJobTest.test_find_executable_task_instances_not_enough_dag_concurrency_for_first-a'
        dag_id_2 = 'SchedulerJobTest.test_find_executable_task_instances_not_enough_dag_concurrency_for_first-b'
        with dag_maker(dag_id=dag_id_1, max_active_tasks=1):
            op1a = EmptyOperator(task_id='dummy1-a', priority_weight=2)
            op1b = EmptyOperator(task_id='dummy1-b', priority_weight=2)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        with dag_maker(dag_id=dag_id_2):
            op2 = EmptyOperator(task_id='dummy2', priority_weight=1)
        dr2 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1a = dr1.get_task_instance(op1a.task_id, session)
        ti1b = dr1.get_task_instance(op1b.task_id, session)
        ti2 = dr2.get_task_instance(op2.task_id, session)
        ti1a.state = State.RUNNING
        ti1b.state = State.SCHEDULED
        ti2.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        assert 1 == len(res)
        assert res[0].key == ti2.key
        session.rollback()

    def test_find_executable_task_instances_not_enough_task_concurrency_for_first(self, dag_maker):
        if False:
            print('Hello World!')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_not_enough_task_concurrency_for_first'
        with dag_maker(dag_id=dag_id):
            op1a = EmptyOperator(task_id='dummy1-a', priority_weight=2, max_active_tis_per_dag=1)
            op1b = EmptyOperator(task_id='dummy1-b', priority_weight=1)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        ti1a = dr1.get_task_instance(op1a.task_id, session)
        ti1b = dr1.get_task_instance(op1b.task_id, session)
        ti2a = dr2.get_task_instance(op1a.task_id, session)
        ti1a.state = State.RUNNING
        ti1b.state = State.SCHEDULED
        ti2a.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        assert 1 == len(res)
        assert res[0].key == ti1b.key
        session.rollback()

    def test_find_executable_task_instances_task_concurrency_per_dagrun_for_first(self, dag_maker):
        if False:
            while True:
                i = 10
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_task_concurrency_per_dagrun_for_first'
        with dag_maker(dag_id=dag_id):
            op1a = EmptyOperator(task_id='dummy1-a', priority_weight=2, max_active_tis_per_dagrun=1)
            op1b = EmptyOperator(task_id='dummy1-b', priority_weight=1)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        ti1a = dr1.get_task_instance(op1a.task_id, session)
        ti1b = dr1.get_task_instance(op1b.task_id, session)
        ti2a = dr2.get_task_instance(op1a.task_id, session)
        ti1a.state = State.RUNNING
        ti1b.state = State.SCHEDULED
        ti2a.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        assert 1 == len(res)
        assert res[0].key == ti2a.key
        session.rollback()

    def test_find_executable_task_instances_not_enough_task_concurrency_per_dagrun_for_first(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_not_enough_task_concurrency_per_dagrun_for_first'
        with dag_maker(dag_id=dag_id):
            op1a = EmptyOperator.partial(task_id='dummy1-a', priority_weight=2, max_active_tis_per_dagrun=1).expand_kwargs([{'inputs': 1}, {'inputs': 2}])
            op1b = EmptyOperator(task_id='dummy1-b', priority_weight=1)
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1a0 = dr.get_task_instance(op1a.task_id, session, map_index=0)
        ti1a1 = dr.get_task_instance(op1a.task_id, session, map_index=1)
        ti1b = dr.get_task_instance(op1b.task_id, session)
        ti1a0.state = State.RUNNING
        ti1a1.state = State.SCHEDULED
        ti1b.state = State.SCHEDULED
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        assert 1 == len(res)
        assert res[0].key == ti1b.key
        session.rollback()

    def test_find_executable_task_instances_negative_open_pool_slots(self, dag_maker):
        if False:
            i = 10
            return i + 15
        '\n        Pools with negative open slots should not block other pools.\n        Negative open slots can happen when reducing the number of total slots in a pool\n        while tasks are running in that pool.\n        '
        set_default_pool_slots(0)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        pool1 = Pool(pool='pool1', slots=1, include_deferred=False)
        pool2 = Pool(pool='pool2', slots=1, include_deferred=False)
        session.add(pool1)
        session.add(pool2)
        dag_id = 'SchedulerJobTest.test_find_executable_task_instances_negative_open_pool_slots'
        with dag_maker(dag_id=dag_id):
            op1 = EmptyOperator(task_id='op1', pool='pool1')
            op2 = EmptyOperator(task_id='op2', pool='pool2', pool_slots=2)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1 = dr1.get_task_instance(op1.task_id, session)
        ti2 = dr1.get_task_instance(op2.task_id, session)
        ti1.state = State.SCHEDULED
        ti2.state = State.RUNNING
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=1, session=session)
        assert 1 == len(res)
        assert res[0].key == ti1.key
        session.rollback()

    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.gauge')
    def test_emit_pool_starving_tasks_metrics(self, mock_stats_gauge, dag_maker):
        if False:
            i = 10
            return i + 15
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dag_id = 'SchedulerJobTest.test_emit_pool_starving_tasks_metrics'
        with dag_maker(dag_id=dag_id):
            op = EmptyOperator(task_id='op', pool_slots=2)
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti = dr.get_task_instance(op.task_id, session)
        ti.state = State.SCHEDULED
        set_default_pool_slots(1)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 0 == len(res)
        mock_stats_gauge.assert_has_calls([mock.call('scheduler.tasks.starving', 1), mock.call(f'pool.starving_tasks.{Pool.DEFAULT_POOL_NAME}', 1), mock.call('pool.starving_tasks', 1, tags={'pool_name': Pool.DEFAULT_POOL_NAME})], any_order=True)
        mock_stats_gauge.reset_mock()
        set_default_pool_slots(2)
        session.flush()
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert 1 == len(res)
        mock_stats_gauge.assert_has_calls([mock.call('scheduler.tasks.starving', 0), mock.call(f'pool.starving_tasks.{Pool.DEFAULT_POOL_NAME}', 0), mock.call('pool.starving_tasks', 0, tags={'pool_name': Pool.DEFAULT_POOL_NAME})], any_order=True)
        session.rollback()
        session.close()

    def test_enqueue_task_instances_with_queued_state(self, dag_maker, session):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_enqueue_task_instances_with_queued_state'
        task_id_1 = 'dummy'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, start_date=DEFAULT_DATE, session=session):
            task1 = EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun()
        ti1 = dr1.get_task_instance(task1.task_id, session)
        with patch.object(BaseExecutor, 'queue_command') as mock_queue_command:
            self.job_runner._enqueue_task_instances_with_queued_state([ti1], session=session)
        assert mock_queue_command.called
        session.rollback()

    @pytest.mark.parametrize('state', [State.FAILED, State.SUCCESS])
    def test_enqueue_task_instances_sets_ti_state_to_None_if_dagrun_in_finish_state(self, state, dag_maker):
        if False:
            print('Hello World!')
        'This tests that task instances whose dagrun is in finished state are not queued'
        dag_id = 'SchedulerJobTest.test_enqueue_task_instances_with_queued_state'
        task_id_1 = 'dummy'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, start_date=DEFAULT_DATE, session=session):
            task1 = EmptyOperator(task_id=task_id_1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun(state=state)
        ti = dr1.get_task_instance(task1.task_id, session)
        ti.state = State.SCHEDULED
        session.merge(ti)
        session.commit()
        with patch.object(BaseExecutor, 'queue_command') as mock_queue_command:
            self.job_runner._enqueue_task_instances_with_queued_state([ti], session=session)
        session.flush()
        ti.refresh_from_db(session=session)
        assert ti.state == State.NONE
        mock_queue_command.assert_not_called()

    def test_critical_section_enqueue_task_instances(self, dag_maker):
        if False:
            i = 10
            return i + 15
        dag_id = 'SchedulerJobTest.test_execute_task_instances'
        task_id_1 = 'dummy_task'
        task_id_2 = 'dummy_task_nonexistent_queue'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, max_active_tasks=3, session=session) as dag:
            task1 = EmptyOperator(task_id=task_id_1)
            task2 = EmptyOperator(task_id=task_id_2)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ti1 = dr1.get_task_instance(task1.task_id, session)
        ti2 = dr1.get_task_instance(task2.task_id, session)
        ti1.state = State.RUNNING
        ti2.state = State.RUNNING
        session.flush()
        assert State.RUNNING == dr1.state
        assert 2 == DAG.get_num_task_instances(dag_id, task_ids=dag.task_ids, states=[State.RUNNING], session=session)
        dr2 = dag_maker.create_dagrun_after(dr1, run_type=DagRunType.SCHEDULED)
        ti3 = dr2.get_task_instance(task1.task_id, session)
        ti4 = dr2.get_task_instance(task2.task_id, session)
        ti3.state = State.SCHEDULED
        ti4.state = State.SCHEDULED
        session.flush()
        assert State.RUNNING == dr2.state
        res = self.job_runner._critical_section_enqueue_task_instances(session)
        ti1.refresh_from_db()
        ti2.refresh_from_db()
        ti3.refresh_from_db()
        ti4.refresh_from_db()
        assert 3 == DAG.get_num_task_instances(dag_id, task_ids=dag.task_ids, states=[State.RUNNING, State.QUEUED], session=session)
        assert State.RUNNING == ti1.state
        assert State.RUNNING == ti2.state
        assert {State.QUEUED, State.SCHEDULED} == {ti3.state, ti4.state}
        assert 1 == res

    def test_execute_task_instances_limit(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'SchedulerJobTest.test_execute_task_instances_limit'
        task_id_1 = 'dummy_task'
        task_id_2 = 'dummy_task_2'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, max_active_tasks=16, session=session):
            task1 = EmptyOperator(task_id=task_id_1)
            task2 = EmptyOperator(task_id=task_id_2)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)

        def _create_dagruns():
            if False:
                i = 10
                return i + 15
            dagrun = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.RUNNING)
            yield dagrun
            for _ in range(3):
                dagrun = dag_maker.create_dagrun_after(dagrun, run_type=DagRunType.SCHEDULED, state=State.RUNNING)
                yield dagrun
        tis = []
        for dr in _create_dagruns():
            ti1 = dr.get_task_instance(task1.task_id, session)
            ti2 = dr.get_task_instance(task2.task_id, session)
            ti1.state = State.SCHEDULED
            ti2.state = State.SCHEDULED
            session.flush()
        scheduler_job.max_tis_per_query = 2
        res = self.job_runner._critical_section_enqueue_task_instances(session)
        assert 2 == res
        scheduler_job.max_tis_per_query = 8
        with mock.patch.object(type(scheduler_job.executor), 'slots_available', new_callable=mock.PropertyMock) as mock_slots:
            mock_slots.return_value = 2
            assert 2 == res
            res = self.job_runner._critical_section_enqueue_task_instances(session)
        res = self.job_runner._critical_section_enqueue_task_instances(session)
        assert 4 == res
        for ti in tis:
            ti.refresh_from_db()
            assert State.QUEUED == ti.state

    def test_execute_task_instances_unlimited(self, dag_maker):
        if False:
            while True:
                i = 10
        'Test that max_tis_per_query=0 is unlimited'
        dag_id = 'SchedulerJobTest.test_execute_task_instances_unlimited'
        task_id_1 = 'dummy_task'
        task_id_2 = 'dummy_task_2'
        session = settings.Session()
        with dag_maker(dag_id=dag_id, max_active_tasks=1024, session=session):
            task1 = EmptyOperator(task_id=task_id_1)
            task2 = EmptyOperator(task_id=task_id_2)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)

        def _create_dagruns():
            if False:
                i = 10
                return i + 15
            dagrun = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.RUNNING)
            yield dagrun
            for _ in range(19):
                dagrun = dag_maker.create_dagrun_after(dagrun, run_type=DagRunType.SCHEDULED, state=State.RUNNING)
                yield dagrun
        for dr in _create_dagruns():
            ti1 = dr.get_task_instance(task1.task_id, session)
            ti2 = dr.get_task_instance(task2.task_id, session)
            ti1.state = State.SCHEDULED
            ti2.state = State.SCHEDULED
            session.flush()
        scheduler_job.max_tis_per_query = 0
        scheduler_job.executor = MagicMock(slots_available=36)
        res = self.job_runner._critical_section_enqueue_task_instances(session)
        assert res == 36
        session.rollback()

    def test_adopt_or_reset_orphaned_tasks(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        session = settings.Session()
        with dag_maker('test_execute_helper_reset_orphaned_tasks') as dag:
            op1 = EmptyOperator(task_id='op1')
        dr = dag_maker.create_dagrun()
        dr2 = dag.create_dagrun(run_type=DagRunType.BACKFILL_JOB, state=State.RUNNING, execution_date=DEFAULT_DATE + datetime.timedelta(1), start_date=DEFAULT_DATE, session=session)
        ti = dr.get_task_instance(task_id=op1.task_id, session=session)
        ti.state = State.QUEUED
        ti2 = dr2.get_task_instance(task_id=op1.task_id, session=session)
        ti2.state = State.QUEUED
        session.commit()
        processor = mock.MagicMock()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=0)
        self.job_runner.processor_agent = processor
        self.job_runner.adopt_or_reset_orphaned_tasks()
        ti = dr.get_task_instance(task_id=op1.task_id, session=session)
        assert ti.state == State.NONE
        ti2 = dr2.get_task_instance(task_id=op1.task_id, session=session)
        assert ti2.state == State.QUEUED, 'Tasks run by Backfill Jobs should not be reset'

    def test_fail_stuck_queued_tasks(self, dag_maker, session):
        if False:
            return 10
        with dag_maker('test_fail_stuck_queued_tasks'):
            op1 = EmptyOperator(task_id='op1')
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance(task_id=op1.task_id, session=session)
        ti.state = State.QUEUED
        ti.queued_dttm = timezone.utcnow() - timedelta(minutes=15)
        session.commit()
        executor = MagicMock()
        executor.cleanup_stuck_queued_tasks = mock.MagicMock()
        scheduler_job = Job(executor=executor)
        job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=0)
        job_runner._task_queued_timeout = 300
        job_runner._fail_tasks_stuck_in_queued()
        job_runner.job.executor.cleanup_stuck_queued_tasks.assert_called_once()

    def test_fail_stuck_queued_tasks_raises_not_implemented(self, dag_maker, session, caplog):
        if False:
            while True:
                i = 10
        with dag_maker('test_fail_stuck_queued_tasks'):
            op1 = EmptyOperator(task_id='op1')
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance(task_id=op1.task_id, session=session)
        ti.state = State.QUEUED
        ti.queued_dttm = timezone.utcnow() - timedelta(minutes=15)
        session.commit()
        from airflow.executors.local_executor import LocalExecutor
        scheduler_job = Job(executor=LocalExecutor())
        job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=0)
        job_runner._task_queued_timeout = 300
        with caplog.at_level(logging.DEBUG):
            job_runner._fail_tasks_stuck_in_queued()
        assert "Executor doesn't support cleanup of stuck queued tasks. Skipping." in caplog.text

    @mock.patch('airflow.dag_processing.manager.DagFileProcessorAgent')
    def test_executor_end_called(self, mock_processor_agent):
        if False:
            print('Hello World!')
        '\n        Test to make sure executor.end gets called with a successful scheduler loop run\n        '
        scheduler_job = Job(executor=mock.MagicMock(slots_available=8))
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull, num_runs=1)
        run_job(scheduler_job, execute_callable=self.job_runner._execute)
        scheduler_job.executor.end.assert_called_once()
        self.job_runner.processor_agent.end.assert_called_once()

    @mock.patch('airflow.dag_processing.manager.DagFileProcessorAgent')
    def test_cleanup_methods_all_called(self, mock_processor_agent):
        if False:
            print('Hello World!')
        '\n        Test to make sure all cleanup methods are called when the scheduler loop has an exception\n        '
        scheduler_job = Job(executor=mock.MagicMock(slots_available=8))
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull, num_runs=1)
        self.job_runner._run_scheduler_loop = mock.MagicMock(side_effect=Exception('oops'))
        mock_processor_agent.return_value.end.side_effect = Exception('double oops')
        scheduler_job.executor.end = mock.MagicMock(side_effect=Exception('triple oops'))
        with pytest.raises(Exception):
            run_job(scheduler_job, execute_callable=self.job_runner._execute)
        self.job_runner.processor_agent.end.assert_called_once()
        scheduler_job.executor.end.assert_called_once()
        mock_processor_agent.return_value.end.reset_mock(side_effect=True)

    def test_queued_dagruns_stops_creating_when_max_active_is_reached(self, dag_maker):
        if False:
            return 10
        'This tests that queued dagruns stops creating once max_active_runs is reached'
        with dag_maker(max_active_runs=10) as dag:
            EmptyOperator(task_id='mytask')
        session = settings.Session()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.processor_agent = mock.MagicMock()
        self.job_runner.dagbag = dag_maker.dagbag
        session = settings.Session()
        orm_dag = session.get(DagModel, dag.dag_id)
        assert orm_dag is not None
        for _ in range(20):
            self.job_runner._create_dag_runs([orm_dag], session)
        drs = session.query(DagRun).all()
        assert len(drs) == 10
        for dr in drs:
            dr.state = State.RUNNING
            session.merge(dr)
        session.commit()
        assert session.query(DagRun.state).filter(DagRun.state == State.RUNNING).count() == 10
        for _ in range(20):
            self.job_runner._create_dag_runs([orm_dag], session)
        assert session.query(DagRun).count() == 10
        assert session.query(DagRun.state).filter(DagRun.state == State.RUNNING).count() == 10
        assert session.query(DagRun.state).filter(DagRun.state == State.QUEUED).count() == 0
        assert orm_dag.next_dagrun_create_after is None

    def test_runs_are_created_after_max_active_runs_was_reached(self, dag_maker, session):
        if False:
            i = 10
            return i + 15
        '\n        Test that when creating runs once max_active_runs is reached the runs does not stick\n        '
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=True)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        with dag_maker(max_active_runs=1, session=session) as dag:
            BashOperator(task_id='task', bash_command='true')
        dag_run = dag_maker.create_dagrun(state=State.RUNNING, session=session, run_type=DagRunType.SCHEDULED)
        for _ in range(3):
            self.job_runner._do_scheduling(session)
        dag_run = session.merge(dag_run)
        session.refresh(dag_run)
        dag_run.get_task_instance(task_id='task', session=session).state = State.SUCCESS
        for _ in range(3):
            self.job_runner._do_scheduling(session)
        dag_runs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(dag_runs) == 2

    def test_dagrun_timeout_verify_max_active_runs(self, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Test if a a dagrun will not be scheduled if max_dag_runs\n        has been reached and dagrun_timeout is not reached\n\n        Test if a a dagrun would be scheduled if max_dag_runs has\n        been reached but dagrun_timeout is also reached\n        '
        with dag_maker(dag_id='test_scheduler_verify_max_active_runs_and_dagrun_timeout', start_date=DEFAULT_DATE, max_active_runs=1, processor_subdir=TEST_DAG_FOLDER, dagrun_timeout=datetime.timedelta(seconds=60)) as dag:
            EmptyOperator(task_id='dummy')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.dagbag = dag_maker.dagbag
        session = settings.Session()
        orm_dag = session.get(DagModel, dag.dag_id)
        assert orm_dag is not None
        self.job_runner._create_dag_runs([orm_dag], session)
        self.job_runner._start_queued_dagruns(session)
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        assert orm_dag.next_dagrun_create_after is None
        assert isinstance(orm_dag.next_dagrun, datetime.datetime)
        assert isinstance(orm_dag.next_dagrun_data_interval_start, datetime.datetime)
        assert isinstance(orm_dag.next_dagrun_data_interval_end, datetime.datetime)
        dr.start_date = timezone.utcnow() - datetime.timedelta(days=1)
        session.flush()
        self.job_runner.processor_agent = mock.Mock()
        callback = self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        session.refresh(dr)
        assert dr.state == State.FAILED
        session.refresh(orm_dag)
        assert isinstance(orm_dag.next_dagrun, datetime.datetime)
        assert isinstance(orm_dag.next_dagrun_data_interval_start, datetime.datetime)
        assert isinstance(orm_dag.next_dagrun_data_interval_end, datetime.datetime)
        assert isinstance(orm_dag.next_dagrun_create_after, datetime.datetime)
        expected_callback = DagCallbackRequest(full_filepath=dr.dag.fileloc, dag_id=dr.dag_id, is_failure_callback=True, run_id=dr.run_id, processor_subdir=TEST_DAG_FOLDER, msg='timed_out')
        assert callback == expected_callback
        session.rollback()
        session.close()

    def test_dagrun_timeout_fails_run(self, dag_maker):
        if False:
            return 10
        '\n        Test if a a dagrun will be set failed if timeout, even without max_active_runs\n        '
        session = settings.Session()
        with dag_maker(dag_id='test_scheduler_fail_dagrun_timeout', dagrun_timeout=datetime.timedelta(seconds=60), processor_subdir=TEST_DAG_FOLDER, session=session):
            EmptyOperator(task_id='dummy')
        dr = dag_maker.create_dagrun(start_date=timezone.utcnow() - datetime.timedelta(days=1))
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.dagbag = dag_maker.dagbag
        self.job_runner.processor_agent = mock.Mock()
        callback = self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        session.refresh(dr)
        assert dr.state == State.FAILED
        expected_callback = DagCallbackRequest(full_filepath=dr.dag.fileloc, dag_id=dr.dag_id, is_failure_callback=True, run_id=dr.run_id, processor_subdir=TEST_DAG_FOLDER, msg='timed_out')
        assert callback == expected_callback
        session.rollback()
        session.close()

    def test_dagrun_timeout_fails_run_and_update_next_dagrun(self, dag_maker):
        if False:
            return 10
        '\n        Test that dagrun timeout fails run and update the next dagrun\n        '
        session = settings.Session()
        with dag_maker(max_active_runs=1, dag_id='test_scheduler_fail_dagrun_timeout', dagrun_timeout=datetime.timedelta(seconds=60)):
            EmptyOperator(task_id='dummy')
        dr = dag_maker.create_dagrun(start_date=timezone.utcnow() - datetime.timedelta(days=1))
        dag_maker.dag_model.next_dagrun == dr.execution_date
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.dagbag = dag_maker.dagbag
        scheduler_job.executor = MockExecutor()
        self.job_runner.processor_agent = mock.Mock()
        self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        session.refresh(dr)
        assert dr.state == State.FAILED
        assert dag_maker.dag_model.next_dagrun_create_after == dr.execution_date + timedelta(days=1)
        assert session.query(DagRun).filter(DagRun.state.in_([DagRunState.RUNNING, DagRunState.QUEUED])).count() == 0

    @pytest.mark.parametrize('state, expected_callback_msg', [(State.SUCCESS, 'success'), (State.FAILED, 'task_failure')])
    def test_dagrun_callbacks_are_called(self, state, expected_callback_msg, dag_maker):
        if False:
            i = 10
            return i + 15
        '\n        Test if DagRun is successful, and if Success callbacks is defined, it is sent to DagFileProcessor.\n        '
        with dag_maker(dag_id='test_dagrun_callbacks_are_called', on_success_callback=lambda x: print('success'), on_failure_callback=lambda x: print('failed'), processor_subdir=TEST_DAG_FOLDER) as dag:
            EmptyOperator(task_id='dummy')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.dagbag = dag_maker.dagbag
        self.job_runner.processor_agent = mock.Mock()
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance('dummy')
        ti.set_state(state, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        expected_callback = DagCallbackRequest(full_filepath=dag.fileloc, dag_id=dr.dag_id, is_failure_callback=bool(state == State.FAILED), run_id=dr.run_id, processor_subdir=TEST_DAG_FOLDER, msg=expected_callback_msg)
        scheduler_job.executor.callback_sink.send.assert_called_once_with(expected_callback)
        session.rollback()
        session.close()

    @pytest.mark.parametrize('state, expected_callback_msg', [(State.SUCCESS, 'success'), (State.FAILED, 'task_failure')])
    def test_dagrun_plugins_are_notified(self, state, expected_callback_msg, dag_maker):
        if False:
            print('Hello World!')
        '\n        Test if DagRun is successful, and if Success callbacks is defined, it is sent to DagFileProcessor.\n        '
        with dag_maker(dag_id='test_dagrun_callbacks_are_called', on_success_callback=lambda x: print('success'), on_failure_callback=lambda x: print('failed'), processor_subdir=TEST_DAG_FOLDER):
            EmptyOperator(task_id='dummy')
        dag_listener.clear()
        get_listener_manager().add_listener(dag_listener)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.dagbag = dag_maker.dagbag
        self.job_runner.processor_agent = mock.Mock()
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance('dummy')
        ti.set_state(state, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        assert len(dag_listener.success) or len(dag_listener.failure)
        dag_listener.success = []
        dag_listener.failure = []
        session.rollback()
        session.close()

    def test_dagrun_timeout_callbacks_are_stored_in_database(self, dag_maker, session):
        if False:
            while True:
                i = 10
        with dag_maker(dag_id='test_dagrun_timeout_callbacks_are_stored_in_database', on_failure_callback=lambda x: print('failed'), dagrun_timeout=timedelta(hours=1), processor_subdir=TEST_DAG_FOLDER) as dag:
            EmptyOperator(task_id='empty')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        scheduler_job.executor.callback_sink = DatabaseCallbackSink()
        self.job_runner.dagbag = dag_maker.dagbag
        self.job_runner.processor_agent = mock.Mock()
        dr = dag_maker.create_dagrun(start_date=DEFAULT_DATE)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        callback = session.query(DbCallbackRequest).order_by(DbCallbackRequest.id.desc()).first().get_callback_request()
        expected_callback = DagCallbackRequest(full_filepath=dag.fileloc, dag_id=dr.dag_id, is_failure_callback=True, run_id=dr.run_id, processor_subdir=TEST_DAG_FOLDER, msg='timed_out')
        assert callback == expected_callback

    def test_dagrun_callbacks_commited_before_sent(self, dag_maker):
        if False:
            print('Hello World!')
        '\n        Tests that before any callbacks are sent to the processor, the session is committed. This ensures\n        that the dagrun details are up to date when the callbacks are run.\n        '
        with dag_maker(dag_id='test_dagrun_callbacks_commited_before_sent'):
            EmptyOperator(task_id='dummy')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.Mock()
        self.job_runner._send_dag_callbacks_to_processor = mock.Mock()
        self.job_runner._schedule_dag_run = mock.Mock()
        dr = dag_maker.create_dagrun()
        session = settings.Session()
        ti = dr.get_task_instance('dummy')
        ti.set_state(State.SUCCESS, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False), mock.patch('airflow.jobs.scheduler_job_runner.prohibit_commit') as mock_guard:
            mock_guard.return_value.__enter__.return_value.commit.side_effect = session.commit

            def mock_schedule_dag_run(*args, **kwargs):
                if False:
                    print('Hello World!')
                mock_guard.reset_mock()
                return None

            def mock_send_dag_callbacks_to_processor(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                mock_guard.return_value.__enter__.return_value.commit.assert_called()
            self.job_runner._send_dag_callbacks_to_processor.side_effect = mock_send_dag_callbacks_to_processor
            self.job_runner._schedule_dag_run.side_effect = mock_schedule_dag_run
            self.job_runner._do_scheduling(session)
        self.job_runner._send_dag_callbacks_to_processor.assert_called_once()
        session.rollback()
        session.close()

    @pytest.mark.parametrize('state', [State.SUCCESS, State.FAILED])
    def test_dagrun_callbacks_are_not_added_when_callbacks_are_not_defined(self, state, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if no on_*_callback are defined on DAG, Callbacks not registered and sent to DAG Processor\n        '
        with dag_maker(dag_id='test_dagrun_callbacks_are_not_added_when_callbacks_are_not_defined'):
            BashOperator(task_id='test_task', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.Mock()
        self.job_runner._send_dag_callbacks_to_processor = mock.Mock()
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance('test_task')
        ti.set_state(state, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        self.job_runner._send_dag_callbacks_to_processor.assert_called_once()
        call_args = self.job_runner._send_dag_callbacks_to_processor.call_args.args
        assert call_args[0].dag_id == dr.dag_id
        assert call_args[1] is None
        session.rollback()
        session.close()

    @pytest.mark.parametrize('state, msg', [[State.SUCCESS, 'success'], [State.FAILED, 'task_failure']])
    def test_dagrun_callbacks_are_added_when_callbacks_are_defined(self, state, msg, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Test if on_*_callback are defined on DAG, Callbacks ARE registered and sent to DAG Processor\n        '
        with dag_maker(dag_id='test_dagrun_callbacks_are_added_when_callbacks_are_defined', on_failure_callback=lambda : True, on_success_callback=lambda : True):
            BashOperator(task_id='test_task', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.Mock()
        self.job_runner._send_dag_callbacks_to_processor = mock.Mock()
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance('test_task')
        ti.set_state(state, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        self.job_runner._send_dag_callbacks_to_processor.assert_called_once()
        call_args = self.job_runner._send_dag_callbacks_to_processor.call_args.args
        assert call_args[0].dag_id == dr.dag_id
        assert call_args[1] is not None
        assert call_args[1].msg == msg
        session.rollback()
        session.close()

    def test_dagrun_notify_called_success(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        with dag_maker(dag_id='test_dagrun_notify_called', on_success_callback=lambda x: print('success'), on_failure_callback=lambda x: print('failed'), processor_subdir=TEST_DAG_FOLDER):
            EmptyOperator(task_id='dummy')
        dag_listener.clear()
        get_listener_manager().add_listener(dag_listener)
        executor = MockExecutor(do_update=False)
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.dagbag = dag_maker.dagbag
        self.job_runner.processor_agent = mock.MagicMock()
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        ti = dr.get_task_instance('dummy')
        ti.set_state(State.SUCCESS, session)
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
        assert dag_listener.success[0].dag_id == dr.dag_id
        assert dag_listener.success[0].run_id == dr.run_id
        assert dag_listener.success[0].state == DagRunState.SUCCESS

    def test_do_not_schedule_removed_task(self, dag_maker):
        if False:
            i = 10
            return i + 15
        schedule_interval = datetime.timedelta(days=1)
        with dag_maker(dag_id='test_scheduler_do_not_schedule_removed_task', schedule=schedule_interval):
            EmptyOperator(task_id='dummy')
        session = settings.Session()
        dr = dag_maker.create_dagrun()
        assert dr is not None
        session.query(DagModel).delete()
        with dag_maker(dag_id='test_scheduler_do_not_schedule_removed_task', schedule=schedule_interval, start_date=DEFAULT_DATE + schedule_interval):
            pass
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        res = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert [] == res

    @provide_session
    def evaluate_dagrun(self, dag_id, expected_task_states, dagrun_state, run_kwargs=None, advance_execution_date=False, session=None):
        if False:
            i = 10
            return i + 15
        '\n        Helper for testing DagRun states with simple two-task DAGs.\n        This is hackish: a dag run is created but its tasks are\n        run by a backfill.\n        '
        if run_kwargs is None:
            run_kwargs = {}
        dag = self.dagbag.get_dag(dag_id)
        dagrun_info = dag.next_dagrun_info(None)
        assert dagrun_info is not None
        dr = dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=dagrun_info.logical_date, state=None, session=session)
        if advance_execution_date:
            dr = dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=dr.data_interval_end, state=None, session=session)
        ex_date = dr.execution_date
        for (tid, state) in expected_task_states.items():
            if state == State.FAILED:
                self.null_exec.mock_task_fail(dag_id, tid, dr.run_id)
        try:
            dag = DagBag().get_dag(dag.dag_id)
            assert not isinstance(dag, SerializedDAG)
            dag.run(start_date=ex_date, end_date=ex_date, executor=self.null_exec, **run_kwargs)
        except AirflowException:
            pass
        dr = DagRun.find(dag_id=dag_id, execution_date=ex_date, session=session)
        dr = dr[0]
        dr.dag = dag
        assert dr.state == dagrun_state
        for (task_id, expected_state) in expected_task_states.items():
            ti = dr.get_task_instance(task_id)
            assert ti.state == expected_state

    def test_dagrun_fail(self):
        if False:
            return 10
        '\n        DagRuns with one failed and one incomplete root task -> FAILED\n        '
        self.evaluate_dagrun(dag_id='test_dagrun_states_fail', expected_task_states={'test_dagrun_fail': State.FAILED, 'test_dagrun_succeed': State.UPSTREAM_FAILED}, dagrun_state=State.FAILED)

    def test_dagrun_success(self):
        if False:
            i = 10
            return i + 15
        '\n        DagRuns with one failed and one successful root task -> SUCCESS\n        '
        self.evaluate_dagrun(dag_id='test_dagrun_states_success', expected_task_states={'test_dagrun_fail': State.FAILED, 'test_dagrun_succeed': State.SUCCESS}, dagrun_state=State.SUCCESS)

    def test_dagrun_root_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        DagRuns with one successful and one failed root task -> FAILED\n        '
        self.evaluate_dagrun(dag_id='test_dagrun_states_root_fail', expected_task_states={'test_dagrun_succeed': State.SUCCESS, 'test_dagrun_fail': State.FAILED}, dagrun_state=State.FAILED)

    def test_dagrun_root_fail_unfinished(self):
        if False:
            return 10
        '\n        DagRuns with one unfinished and one failed root task -> RUNNING\n        '
        dag_id = 'test_dagrun_states_root_fail_unfinished'
        dag = self.dagbag.get_dag(dag_id)
        dr = dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=DEFAULT_DATE, state=None)
        self.null_exec.mock_task_fail(dag_id, 'test_dagrun_fail', dr.run_id)
        with pytest.raises(AirflowException):
            dag.run(start_date=dr.execution_date, end_date=dr.execution_date, executor=self.null_exec)
        with create_session() as session:
            ti = dr.get_task_instance('test_dagrun_unfinished', session=session)
            ti.state = State.NONE
            session.commit()
        dr.update_state()
        assert dr.state == State.RUNNING

    def test_dagrun_root_after_dagrun_unfinished(self):
        if False:
            print('Hello World!')
        '\n        DagRuns with one successful and one future root task -> SUCCESS\n\n        Noted: the DagRun state could be still in running state during CI.\n        '
        clear_db_dags()
        dag_id = 'test_dagrun_states_root_future'
        dag = self.dagbag.get_dag(dag_id)
        dag.sync_to_db()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=1, subdir=dag.fileloc)
        run_job(scheduler_job, execute_callable=self.job_runner._execute)
        first_run = DagRun.find(dag_id=dag_id, execution_date=DEFAULT_DATE)[0]
        ti_ids = [(ti.task_id, ti.state) for ti in first_run.get_task_instances()]
        assert ti_ids == [('current', State.SUCCESS)]
        assert first_run.state in [State.SUCCESS, State.RUNNING]

    def test_dagrun_deadlock_ignore_depends_on_past_advance_ex_date(self):
        if False:
            return 10
        '\n        DagRun is marked a success if ignore_first_depends_on_past=True\n\n        Test that an otherwise-deadlocked dagrun is marked as a success\n        if ignore_first_depends_on_past=True and the dagrun execution_date\n        is after the start_date.\n        '
        self.evaluate_dagrun(dag_id='test_dagrun_states_deadlock', expected_task_states={'test_depends_on_past': State.SUCCESS, 'test_depends_on_past_2': State.SUCCESS}, dagrun_state=State.SUCCESS, advance_execution_date=True, run_kwargs=dict(ignore_first_depends_on_past=True))

    def test_dagrun_deadlock_ignore_depends_on_past(self):
        if False:
            print('Hello World!')
        "\n        Test that ignore_first_depends_on_past doesn't affect results\n        (this is the same test as\n        test_dagrun_deadlock_ignore_depends_on_past_advance_ex_date except\n        that start_date == execution_date so depends_on_past is irrelevant).\n        "
        self.evaluate_dagrun(dag_id='test_dagrun_states_deadlock', expected_task_states={'test_depends_on_past': State.SUCCESS, 'test_depends_on_past_2': State.SUCCESS}, dagrun_state=State.SUCCESS, run_kwargs=dict(ignore_first_depends_on_past=True))

    @pytest.mark.parametrize('configs', [{('scheduler', 'standalone_dag_processor'): 'False'}, {('scheduler', 'standalone_dag_processor'): 'True'}])
    def test_scheduler_start_date(self, configs):
        if False:
            while True:
                i = 10
        '\n        Test that the scheduler respects start_dates, even when DAGs have run\n        '
        with conf_vars(configs):
            with create_session() as session:
                dag_id = 'test_start_date_scheduling'
                dag = self.dagbag.get_dag(dag_id)
                dag.clear()
                assert dag.start_date > datetime.datetime.now(timezone.utc)
                other_dag = self.dagbag.get_dag('test_task_start_date_scheduling')
                other_dag.is_paused_upon_creation = True
                other_dag.sync_to_db()
                scheduler_job = Job(executor=self.null_exec)
                self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=dag.fileloc, num_runs=1)
                run_job(scheduler_job, execute_callable=self.job_runner._execute)
                assert len(session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id).all()) == 0
                session.commit()
                assert [] == self.null_exec.sorted_tasks
                bf_exec = MockExecutor()
                backfill_job = Job(executor=bf_exec)
                job_runner = BackfillJobRunner(job=backfill_job, dag=dag, start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
                run_job(job=backfill_job, execute_callable=job_runner._execute)
                assert len(session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id).all()) == 1
                assert [(TaskInstanceKey(dag.dag_id, 'dummy', f'backfill__{DEFAULT_DATE.isoformat()}', 1), (State.SUCCESS, None))] == bf_exec.sorted_tasks
                session.commit()
                scheduler_job = Job(executor=self.null_exec)
                self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=dag.fileloc, num_runs=1)
                run_job(scheduler_job, execute_callable=self.job_runner._execute)
                assert len(session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id).all()) == 1
                session.commit()
                assert [] == self.null_exec.sorted_tasks

    @pytest.mark.parametrize('configs', [{('scheduler', 'standalone_dag_processor'): 'False'}, {('scheduler', 'standalone_dag_processor'): 'True'}])
    def test_scheduler_task_start_date(self, configs):
        if False:
            i = 10
            return i + 15
        '\n        Test that the scheduler respects task start dates that are different from DAG start dates\n        '
        with conf_vars(configs):
            dagbag = DagBag(dag_folder=os.path.join(settings.DAGS_FOLDER, 'test_scheduler_dags.py'), include_examples=False)
            dag_id = 'test_task_start_date_scheduling'
            dag = self.dagbag.get_dag(dag_id)
            dag.is_paused_upon_creation = False
            dagbag.bag_dag(dag=dag, root_dag=dag)
            other_dag = self.dagbag.get_dag('test_start_date_scheduling')
            other_dag.is_paused_upon_creation = True
            dagbag.bag_dag(dag=other_dag, root_dag=other_dag)
            dagbag.sync_to_db()
            scheduler_job = Job(executor=self.null_exec)
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=dag.fileloc, num_runs=3)
            run_job(scheduler_job, execute_callable=self.job_runner._execute)
            session = settings.Session()
            tiq = session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id)
            ti1s = tiq.filter(TaskInstance.task_id == 'dummy1').all()
            ti2s = tiq.filter(TaskInstance.task_id == 'dummy2').all()
            assert len(ti1s) == 0
            assert len(ti2s) >= 2
            for task in ti2s:
                assert task.state == State.SUCCESS

    @pytest.mark.parametrize('configs', [{('scheduler', 'standalone_dag_processor'): 'False'}, {('scheduler', 'standalone_dag_processor'): 'True'}])
    def test_scheduler_multiprocessing(self, configs):
        if False:
            return 10
        '\n        Test that the scheduler can successfully queue multiple dags in parallel\n        '
        with conf_vars(configs):
            dag_ids = ['test_start_date_scheduling', 'test_dagrun_states_success']
            for dag_id in dag_ids:
                dag = self.dagbag.get_dag(dag_id)
                dag.clear()
            scheduler_job = Job(executor=self.null_exec)
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.path.join(TEST_DAG_FOLDER, 'test_scheduler_dags.py'), num_runs=1)
            run_job(scheduler_job, execute_callable=self.job_runner._execute)
            dag_id = 'test_start_date_scheduling'
            session = settings.Session()
            assert len(session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id).all()) == 0

    @pytest.mark.parametrize('configs', [{('scheduler', 'standalone_dag_processor'): 'False'}, {('scheduler', 'standalone_dag_processor'): 'True'}])
    def test_scheduler_verify_pool_full(self, dag_maker, configs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test task instances not queued when pool is full\n        '
        with conf_vars(configs):
            with dag_maker(dag_id='test_scheduler_verify_pool_full'):
                BashOperator(task_id='dummy', pool='test_scheduler_verify_pool_full', bash_command='echo hi')
            session = settings.Session()
            pool = Pool(pool='test_scheduler_verify_pool_full', slots=1, include_deferred=False)
            session.add(pool)
            session.flush()
            scheduler_job = Job(executor=self.null_exec)
            self.job_runner = SchedulerJobRunner(job=scheduler_job)
            self.job_runner.processor_agent = mock.MagicMock()
            dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
            self.job_runner._schedule_dag_run(dr, session)
            dr = dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED, state=State.RUNNING)
            self.job_runner._schedule_dag_run(dr, session)
            session.flush()
            task_instances_list = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
            assert len(task_instances_list) == 1

    @pytest.mark.need_serialized_dag
    def test_scheduler_verify_pool_full_2_slots_per_task(self, dag_maker, session):
        if False:
            i = 10
            return i + 15
        '\n        Test task instances not queued when pool is full.\n\n        Variation with non-default pool_slots\n        '
        with dag_maker(dag_id='test_scheduler_verify_pool_full_2_slots_per_task', start_date=DEFAULT_DATE, session=session):
            BashOperator(task_id='dummy', pool='test_scheduler_verify_pool_full_2_slots_per_task', pool_slots=2, bash_command='echo hi')
        pool = Pool(pool='test_scheduler_verify_pool_full_2_slots_per_task', slots=6, include_deferred=False)
        session.add(pool)
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()

        def _create_dagruns():
            if False:
                i = 10
                return i + 15
            dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
            yield dr
            for _ in range(4):
                dr = dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED)
                yield dr
        for dr in _create_dagruns():
            self.job_runner._schedule_dag_run(dr, session)
        task_instances_list = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert len(task_instances_list) == 3

    def test_scheduler_keeps_scheduling_pool_full(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test task instances in a pool that isn't full keep getting scheduled even when a pool is full.\n        "
        with dag_maker(dag_id='test_scheduler_keeps_scheduling_pool_full_d1', start_date=DEFAULT_DATE):
            BashOperator(task_id='test_scheduler_keeps_scheduling_pool_full_t1', pool='test_scheduler_keeps_scheduling_pool_full_p1', bash_command='echo hi')
        dag_d1 = dag_maker.dag
        with dag_maker(dag_id='test_scheduler_keeps_scheduling_pool_full_d2', start_date=DEFAULT_DATE):
            BashOperator(task_id='test_scheduler_keeps_scheduling_pool_full_t2', pool='test_scheduler_keeps_scheduling_pool_full_p2', bash_command='echo hi')
        dag_d2 = dag_maker.dag
        session = settings.Session()
        pool_p1 = Pool(pool='test_scheduler_keeps_scheduling_pool_full_p1', slots=1, include_deferred=False)
        pool_p2 = Pool(pool='test_scheduler_keeps_scheduling_pool_full_p2', slots=10, include_deferred=False)
        session.add(pool_p1)
        session.add(pool_p2)
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()

        def _create_dagruns(dag: DAG):
            if False:
                return 10
            next_info = dag.next_dagrun_info(None)
            assert next_info is not None
            for _ in range(30):
                yield dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=next_info.logical_date, data_interval=next_info.data_interval, state=DagRunState.RUNNING)
                next_info = dag.next_dagrun_info(next_info.data_interval)
                if next_info is None:
                    break
        for dr in _create_dagruns(dag_d1):
            self.job_runner._schedule_dag_run(dr, session)
        for dr in _create_dagruns(dag_d2):
            self.job_runner._schedule_dag_run(dr, session)
        self.job_runner._executable_task_instances_to_queued(max_tis=2, session=session)
        task_instances_list2 = self.job_runner._executable_task_instances_to_queued(max_tis=2, session=session)
        assert len(task_instances_list2) > 0
        assert all((task_instance.pool != 'test_scheduler_keeps_scheduling_pool_full_p1' for task_instance in task_instances_list2))

    def test_scheduler_verify_priority_and_slots(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test task instances with higher priority are not queued\n        when pool does not have enough slots.\n\n        Though tasks with lower priority might be executed.\n        '
        with dag_maker(dag_id='test_scheduler_verify_priority_and_slots'):
            BashOperator(task_id='test_scheduler_verify_priority_and_slots_t0', pool='test_scheduler_verify_priority_and_slots', pool_slots=2, priority_weight=2, bash_command='echo hi')
            BashOperator(task_id='test_scheduler_verify_priority_and_slots_t1', pool='test_scheduler_verify_priority_and_slots', pool_slots=1, priority_weight=3, bash_command='echo hi')
            BashOperator(task_id='test_scheduler_verify_priority_and_slots_t2', pool='test_scheduler_verify_priority_and_slots', pool_slots=1, priority_weight=1, bash_command='echo hi')
        session = settings.Session()
        pool = Pool(pool='test_scheduler_verify_priority_and_slots', slots=2, include_deferred=False)
        session.add(pool)
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        dr = dag_maker.create_dagrun()
        for ti in dr.task_instances:
            ti.state = State.SCHEDULED
            session.merge(ti)
        session.flush()
        task_instances_list = self.job_runner._executable_task_instances_to_queued(max_tis=32, session=session)
        assert len(task_instances_list) == 2
        ti0 = session.query(TaskInstance).filter(TaskInstance.task_id == 'test_scheduler_verify_priority_and_slots_t0').first()
        assert ti0.state == State.SCHEDULED
        ti1 = session.query(TaskInstance).filter(TaskInstance.task_id == 'test_scheduler_verify_priority_and_slots_t1').first()
        assert ti1.state == State.QUEUED
        ti2 = session.query(TaskInstance).filter(TaskInstance.task_id == 'test_scheduler_verify_priority_and_slots_t2').first()
        assert ti2.state == State.QUEUED

    def test_verify_integrity_if_dag_not_changed(self, dag_maker):
        if False:
            while True:
                i = 10
        with create_session() as session:
            session.query(SerializedDagModel).filter(SerializedDagModel.dag_id == 'test_verify_integrity_if_dag_not_changed').delete(synchronize_session=False)
        with dag_maker(dag_id='test_verify_integrity_if_dag_not_changed') as dag:
            BashOperator(task_id='dummy', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        orm_dag = dag_maker.dag_model
        assert orm_dag is not None
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dag = self.job_runner.dagbag.get_dag('test_verify_integrity_if_dag_not_changed', session=session)
        self.job_runner._create_dag_runs([orm_dag], session)
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        with mock.patch('airflow.jobs.scheduler_job_runner.DagRun.verify_integrity') as mock_verify_integrity:
            self.job_runner._schedule_dag_run(dr, session)
            mock_verify_integrity.assert_not_called()
        session.flush()
        tis_count = session.query(func.count(TaskInstance.task_id)).filter(TaskInstance.dag_id == dr.dag_id, TaskInstance.execution_date == dr.execution_date, TaskInstance.task_id == dr.dag.tasks[0].task_id, TaskInstance.state == State.SCHEDULED).scalar()
        assert tis_count == 1
        latest_dag_version = SerializedDagModel.get_latest_version_hash(dr.dag_id, session=session)
        assert dr.dag_hash == latest_dag_version
        session.rollback()
        session.close()

    def test_verify_integrity_if_dag_changed(self, dag_maker):
        if False:
            i = 10
            return i + 15
        with create_session() as session:
            session.query(SerializedDagModel).filter(SerializedDagModel.dag_id == 'test_verify_integrity_if_dag_changed').delete(synchronize_session=False)
        with dag_maker(dag_id='test_verify_integrity_if_dag_changed') as dag:
            BashOperator(task_id='dummy', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        orm_dag = dag_maker.dag_model
        assert orm_dag is not None
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dag = self.job_runner.dagbag.get_dag('test_verify_integrity_if_dag_changed', session=session)
        self.job_runner._create_dag_runs([orm_dag], session)
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        dag_version_1 = SerializedDagModel.get_latest_version_hash(dr.dag_id, session=session)
        assert dr.dag_hash == dag_version_1
        assert self.job_runner.dagbag.dags == {'test_verify_integrity_if_dag_changed': dag}
        assert len(self.job_runner.dagbag.dags.get('test_verify_integrity_if_dag_changed').tasks) == 1
        BashOperator(task_id='bash_task_1', dag=dag, bash_command='echo hi')
        SerializedDagModel.write_dag(dag=dag)
        dag_version_2 = SerializedDagModel.get_latest_version_hash(dr.dag_id, session=session)
        assert dag_version_2 != dag_version_1
        self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        assert dr.dag_hash == dag_version_2
        assert self.job_runner.dagbag.dags == {'test_verify_integrity_if_dag_changed': dag}
        assert len(self.job_runner.dagbag.dags.get('test_verify_integrity_if_dag_changed').tasks) == 2
        tis_count = session.query(func.count(TaskInstance.task_id)).filter(TaskInstance.dag_id == dr.dag_id, TaskInstance.execution_date == dr.execution_date, TaskInstance.state == State.SCHEDULED).scalar()
        assert tis_count == 2
        latest_dag_version = SerializedDagModel.get_latest_version_hash(dr.dag_id, session=session)
        assert dr.dag_hash == latest_dag_version
        session.rollback()
        session.close()

    def test_verify_integrity_if_dag_disappeared(self, dag_maker, caplog):
        if False:
            return 10
        with create_session() as session:
            session.query(SerializedDagModel).filter(SerializedDagModel.dag_id == 'test_verify_integrity_if_dag_disappeared').delete(synchronize_session=False)
        with dag_maker(dag_id='test_verify_integrity_if_dag_disappeared') as dag:
            BashOperator(task_id='dummy', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        orm_dag = dag_maker.dag_model
        assert orm_dag is not None
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dag = self.job_runner.dagbag.get_dag('test_verify_integrity_if_dag_disappeared', session=session)
        self.job_runner._create_dag_runs([orm_dag], session)
        dag_id = dag.dag_id
        drs = DagRun.find(dag_id=dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        dag_version_1 = SerializedDagModel.get_latest_version_hash(dag_id, session=session)
        assert dr.dag_hash == dag_version_1
        assert self.job_runner.dagbag.dags == {'test_verify_integrity_if_dag_disappeared': dag}
        assert len(self.job_runner.dagbag.dags.get('test_verify_integrity_if_dag_disappeared').tasks) == 1
        SerializedDagModel.remove_dag(dag_id=dag_id)
        dag = self.job_runner.dagbag.dags[dag_id]
        self.job_runner.dagbag.dags = MagicMock()
        self.job_runner.dagbag.dags.get.side_effect = [dag, None]
        session.flush()
        with caplog.at_level(logging.WARNING):
            callback = self.job_runner._schedule_dag_run(dr, session)
            assert 'The DAG disappeared before verifying integrity' in caplog.text
        assert callback is None
        session.rollback()
        session.close()

    @pytest.mark.need_serialized_dag
    def test_retry_still_in_executor(self, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Checks if the scheduler does not put a task in limbo, when a task is retried\n        but is still present in the executor.\n        '
        executor = MockExecutor(do_update=False)
        with create_session() as session:
            with dag_maker(dag_id='test_retry_still_in_executor', schedule='@once', session=session):
                dag_task1 = BashOperator(task_id='test_retry_handling_op', bash_command='exit 1', retries=1)
            dag_maker.dag_model.calculate_dagrun_date_fields(dag_maker.dag, None)

        @provide_session
        def do_schedule(session):
            if False:
                print('Hello World!')
            scheduler_job = Job(executor=executor)
            self.job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=1, subdir=os.devnull)
            self.job_runner.dagbag = dag_maker.dagbag
            scheduler_job.heartrate = 0
            with mock.patch.object(DagModel, 'deactivate_deleted_dags'):
                run_job(scheduler_job, execute_callable=self.job_runner._execute)
        do_schedule()
        with create_session() as session:
            ti = session.query(TaskInstance).filter(TaskInstance.dag_id == 'test_retry_still_in_executor', TaskInstance.task_id == 'test_retry_handling_op').first()
        assert ti is not None, 'Task not created by scheduler'
        ti.task = dag_task1

        def run_with_error(ti, ignore_ti_state=False):
            if False:
                return 10
            with contextlib.suppress(AirflowException):
                ti.run(ignore_ti_state=ignore_ti_state)
        assert ti.try_number == 1
        run_with_error(ti, ignore_ti_state=True)
        assert ti.state == State.UP_FOR_RETRY
        assert ti.try_number == 2
        with create_session() as session:
            ti.refresh_from_db(lock_for_update=True, session=session)
            ti.state = State.SCHEDULED
            session.merge(ti)
        executor.do_update = True
        do_schedule()
        ti.refresh_from_db()
        assert ti.state == State.SUCCESS

    def test_retry_handling_job(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Integration test of the scheduler not accidentally resetting\n        the try_numbers for a task\n        '
        dag = self.dagbag.get_dag('test_retry_handling_job')
        dag_task1 = dag.get_task('test_retry_handling_op')
        dag.clear()
        dag.sync_to_db()
        scheduler_job = Job(job_type=SchedulerJobRunner.job_type, heartrate=0)
        self.job_runner = SchedulerJobRunner(job=scheduler_job, num_runs=1)
        self.job_runner.processor_agent = mock.MagicMock()
        run_job(scheduler_job, execute_callable=self.job_runner._execute)
        session = settings.Session()
        ti = session.query(TaskInstance).filter(TaskInstance.dag_id == dag.dag_id, TaskInstance.task_id == dag_task1.task_id).first()
        assert ti.try_number == 2
        assert ti.state == State.UP_FOR_RETRY

    def test_dag_get_active_runs(self, dag_maker):
        if False:
            i = 10
            return i + 15
        '\n        Test to check that a DAG returns its active runs\n        '
        now = timezone.utcnow()
        six_hours_ago_to_the_hour = (now - datetime.timedelta(hours=6)).replace(minute=0, second=0, microsecond=0)
        start_date = six_hours_ago_to_the_hour
        dag_name1 = 'get_active_runs_test'
        default_args = {'depends_on_past': False, 'start_date': start_date}
        with dag_maker(dag_name1, schedule='* * * * *', max_active_runs=1, default_args=default_args) as dag1:
            run_this_1 = EmptyOperator(task_id='run_this_1')
            run_this_2 = EmptyOperator(task_id='run_this_2')
            run_this_2.set_upstream(run_this_1)
            run_this_3 = EmptyOperator(task_id='run_this_3')
            run_this_3.set_upstream(run_this_2)
        dr = dag_maker.create_dagrun()
        assert dr is not None
        execution_date = dr.execution_date
        running_dates = dag1.get_active_runs()
        try:
            running_date = running_dates[0]
        except Exception:
            running_date = 'Except'
        assert execution_date == running_date, 'Running Date must match Execution Date'

    def test_list_py_file_paths(self):
        if False:
            return 10
        "\n        [JIRA-1357] Test the 'list_py_file_paths' function used by the\n        scheduler to list and load DAGs.\n        "
        detected_files = set()
        expected_files = set()
        ignored_files = {'no_dags.py', 'test_invalid_cron.py', 'test_invalid_dup_task.py', 'test_ignore_this.py', 'test_invalid_param.py', 'test_invalid_param2.py', 'test_invalid_param3.py', 'test_invalid_param4.py', 'test_nested_dag.py', 'test_imports.py', '__init__.py'}
        for (root, _, files) in os.walk(TEST_DAG_FOLDER):
            for file_name in files:
                if file_name.endswith(('.py', '.zip')):
                    if file_name not in ignored_files:
                        expected_files.add(f'{root}/{file_name}')
        for file_path in list_py_file_paths(TEST_DAG_FOLDER, include_examples=False):
            detected_files.add(file_path)
        assert detected_files == expected_files
        ignored_files = {'helper.py'}
        example_dag_folder = airflow.example_dags.__path__[0]
        for (root, _, files) in os.walk(example_dag_folder):
            for file_name in files:
                if file_name.endswith(('.py', '.zip')):
                    if file_name not in ['__init__.py'] and file_name not in ignored_files:
                        expected_files.add(os.path.join(root, file_name))
        detected_files.clear()
        for file_path in list_py_file_paths(TEST_DAG_FOLDER, include_examples=True):
            detected_files.add(file_path)
        assert detected_files == expected_files

    def test_adopt_or_reset_orphaned_tasks_nothing(self):
        if False:
            for i in range(10):
                print('nop')
        'Try with nothing.'
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        session = settings.Session()
        assert 0 == self.job_runner.adopt_or_reset_orphaned_tasks(session=session)

    @pytest.mark.parametrize('adoptable_state', list(sorted(State.adoptable_states)))
    def test_adopt_or_reset_resettable_tasks(self, dag_maker, adoptable_state):
        if False:
            return 10
        dag_id = 'test_adopt_or_reset_adoptable_tasks_' + adoptable_state.name
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            task_id = dag_id + '_task'
            EmptyOperator(task_id=task_id)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(external_trigger=True)
        ti = dr1.get_task_instances(session=session)[0]
        ti.state = adoptable_state
        session.merge(ti)
        session.merge(dr1)
        session.commit()
        num_reset_tis = self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        assert 1 == num_reset_tis

    def test_adopt_or_reset_orphaned_tasks_external_triggered_dag(self, dag_maker):
        if False:
            return 10
        dag_id = 'test_reset_orphaned_tasks_external_triggered_dag'
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            task_id = dag_id + '_task'
            EmptyOperator(task_id=task_id)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        dr1 = dag_maker.create_dagrun(external_trigger=True)
        ti = dr1.get_task_instances(session=session)[0]
        ti.state = State.QUEUED
        session.merge(ti)
        session.merge(dr1)
        session.commit()
        num_reset_tis = self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        assert 1 == num_reset_tis

    def test_adopt_or_reset_orphaned_tasks_backfill_dag(self, dag_maker):
        if False:
            while True:
                i = 10
        dag_id = 'test_adopt_or_reset_orphaned_tasks_backfill_dag'
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            task_id = dag_id + '_task'
            EmptyOperator(task_id=task_id)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        session.add(scheduler_job)
        session.flush()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.BACKFILL_JOB)
        ti = dr1.get_task_instances(session=session)[0]
        ti.state = State.SCHEDULED
        session.merge(ti)
        session.merge(dr1)
        session.flush()
        assert dr1.is_backfill
        assert 0 == self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        session.rollback()

    def test_reset_orphaned_tasks_no_orphans(self, dag_maker):
        if False:
            i = 10
            return i + 15
        dag_id = 'test_reset_orphaned_tasks_no_orphans'
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            task_id = dag_id + '_task'
            EmptyOperator(task_id=task_id)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        session.add(scheduler_job)
        session.flush()
        dr1 = dag_maker.create_dagrun()
        tis = dr1.get_task_instances(session=session)
        tis[0].state = State.RUNNING
        tis[0].queued_by_job_id = scheduler_job.id
        session.merge(dr1)
        session.merge(tis[0])
        session.flush()
        assert 0 == self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        tis[0].refresh_from_db()
        assert State.RUNNING == tis[0].state

    def test_reset_orphaned_tasks_non_running_dagruns(self, dag_maker):
        if False:
            return 10
        'Ensure orphaned tasks with non-running dagruns are not reset.'
        dag_id = 'test_reset_orphaned_tasks_non_running_dagruns'
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            task_id = dag_id + '_task'
            EmptyOperator(task_id=task_id)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        session.add(scheduler_job)
        session.flush()
        dr1 = dag_maker.create_dagrun()
        dr1.state = State.QUEUED
        tis = dr1.get_task_instances(session=session)
        assert 1 == len(tis)
        tis[0].state = State.SCHEDULED
        session.merge(dr1)
        session.merge(tis[0])
        session.flush()
        assert 0 == self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        session.rollback()

    def test_adopt_or_reset_orphaned_tasks_stale_scheduler_jobs(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'test_adopt_or_reset_orphaned_tasks_stale_scheduler_jobs'
        with dag_maker(dag_id=dag_id, schedule='@daily'):
            EmptyOperator(task_id='task1')
            EmptyOperator(task_id='task2')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        session = settings.Session()
        scheduler_job.state = State.RUNNING
        scheduler_job.latest_heartbeat = timezone.utcnow()
        session.add(scheduler_job)
        old_job = Job()
        old_job_runner = SchedulerJobRunner(job=old_job, subdir=os.devnull)
        old_job.state = State.RUNNING
        old_job.latest_heartbeat = timezone.utcnow() - timedelta(minutes=15)
        session.add(old_job)
        session.flush()
        dr1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=DEFAULT_DATE, start_date=timezone.utcnow())
        (ti1, ti2) = dr1.get_task_instances(session=session)
        dr1.state = State.RUNNING
        ti1.state = State.QUEUED
        ti1.queued_by_job_id = old_job.id
        session.merge(dr1)
        session.merge(ti1)
        ti2.state = State.QUEUED
        ti2.queued_by_job_id = scheduler_job.id
        session.merge(ti2)
        session.flush()
        num_reset_tis = self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        assert 1 == num_reset_tis
        session.refresh(ti1)
        assert ti1.state is None
        session.refresh(ti2)
        assert ti2.state == State.QUEUED
        session.rollback()
        if old_job_runner.processor_agent:
            old_job_runner.processor_agent.end()

    def test_adopt_or_reset_orphaned_tasks_only_fails_scheduler_jobs(self, caplog):
        if False:
            print('Hello World!')
        'Make sure we only set SchedulerJobs to failed, not all jobs'
        session = settings.Session()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.state = State.RUNNING
        scheduler_job.latest_heartbeat = timezone.utcnow()
        session.add(scheduler_job)
        session.flush()
        old_job = Job()
        self.job_runner = SchedulerJobRunner(job=old_job, subdir=os.devnull)
        old_job.state = State.RUNNING
        old_job.latest_heartbeat = timezone.utcnow() - timedelta(minutes=15)
        session.add(old_job)
        session.flush()
        old_task_job = Job(state=State.RUNNING)
        old_task_job.latest_heartbeat = timezone.utcnow() - timedelta(minutes=15)
        session.add(old_task_job)
        session.flush()
        with caplog.at_level('INFO', logger='airflow.jobs.scheduler_job_runner'):
            self.job_runner.adopt_or_reset_orphaned_tasks(session=session)
        session.expire_all()
        assert old_job.state == State.FAILED
        assert old_task_job.state == State.RUNNING
        assert 'Marked 1 SchedulerJob instances as failed' in caplog.messages

    def test_send_sla_callbacks_to_processor_sla_disabled(self, dag_maker):
        if False:
            while True:
                i = 10
        'Test SLA Callbacks are not sent when check_slas is False'
        dag_id = 'test_send_sla_callbacks_to_processor_sla_disabled'
        with dag_maker(dag_id=dag_id, schedule='@daily') as dag:
            EmptyOperator(task_id='task1')
        with patch.object(settings, 'CHECK_SLAS', False):
            scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            scheduler_job.executor = MockExecutor()
            self.job_runner._send_sla_callbacks_to_processor(dag)
            scheduler_job.executor.callback_sink.send.assert_not_called()

    def test_send_sla_callbacks_to_processor_sla_no_task_slas(self, dag_maker):
        if False:
            return 10
        'Test SLA Callbacks are not sent when no task SLAs are defined'
        dag_id = 'test_send_sla_callbacks_to_processor_sla_no_task_slas'
        with dag_maker(dag_id=dag_id, schedule='@daily') as dag:
            EmptyOperator(task_id='task1')
        with patch.object(settings, 'CHECK_SLAS', True):
            scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            scheduler_job.executor = MockExecutor()
            self.job_runner._send_sla_callbacks_to_processor(dag)
            scheduler_job.executor.callback_sink.send.assert_not_called()

    @pytest.mark.parametrize('schedule', ['@daily', '0 10 * * *', timedelta(hours=2)])
    def test_send_sla_callbacks_to_processor_sla_with_task_slas(self, schedule, dag_maker):
        if False:
            i = 10
            return i + 15
        'Test SLA Callbacks are sent to the DAG Processor when SLAs are defined on tasks'
        dag_id = 'test_send_sla_callbacks_to_processor_sla_with_task_slas'
        with dag_maker(dag_id=dag_id, schedule=schedule, processor_subdir=TEST_DAG_FOLDER) as dag:
            EmptyOperator(task_id='task1', sla=timedelta(seconds=60))
        with patch.object(settings, 'CHECK_SLAS', True):
            scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            scheduler_job.executor = MockExecutor()
            self.job_runner._send_sla_callbacks_to_processor(dag)
            expected_callback = SlaCallbackRequest(full_filepath=dag.fileloc, dag_id=dag.dag_id, processor_subdir=TEST_DAG_FOLDER)
            scheduler_job.executor.callback_sink.send.assert_called_once_with(expected_callback)

    @pytest.mark.parametrize('schedule', [None, [Dataset('foo')]])
    def test_send_sla_callbacks_to_processor_sla_dag_not_scheduled(self, schedule, dag_maker):
        if False:
            while True:
                i = 10
        "Test SLA Callbacks are not sent when DAG isn't scheduled"
        dag_id = 'test_send_sla_callbacks_to_processor_sla_no_task_slas'
        with dag_maker(dag_id=dag_id, schedule=schedule) as dag:
            EmptyOperator(task_id='task1', sla=timedelta(seconds=5))
        with patch.object(settings, 'CHECK_SLAS', True):
            scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            scheduler_job.executor = MockExecutor()
            self.job_runner._send_sla_callbacks_to_processor(dag)
            scheduler_job.executor.callback_sink.send.assert_not_called()

    @pytest.mark.parametrize('schedule, number_running, excepted', [(None, None, False), ('*/1 * * * *', None, False), ('*/1 * * * *', 1, True)], ids=['no_dag_schedule', 'dag_schedule_too_many_runs', 'dag_schedule_less_runs'])
    def test_should_update_dag_next_dagruns(self, schedule, number_running, excepted, session, dag_maker):
        if False:
            while True:
                i = 10
        'Test if really required to update next dagrun or possible to save run time'
        with dag_maker(dag_id='test_should_update_dag_next_dagruns', schedule=schedule, max_active_runs=2) as dag:
            EmptyOperator(task_id='dummy')
        dag_model = dag_maker.dag_model
        for index in range(2):
            dag_maker.create_dagrun(run_id=f'run_{index}', execution_date=DEFAULT_DATE + timedelta(days=index), start_date=timezone.utcnow(), state=State.RUNNING, session=session)
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        assert excepted is self.job_runner._should_update_dag_next_dagruns(dag, dag_model, total_active_runs=number_running, session=session)

    @pytest.mark.parametrize('run_type, should_update', [(DagRunType.MANUAL, False), (DagRunType.SCHEDULED, True), (DagRunType.BACKFILL_JOB, True), (DagRunType.DATASET_TRIGGERED, False)], ids=[DagRunType.MANUAL.name, DagRunType.SCHEDULED.name, DagRunType.BACKFILL_JOB.name, DagRunType.DATASET_TRIGGERED.name])
    def test_should_update_dag_next_dagruns_after_run_type(self, run_type, should_update, session, dag_maker):
        if False:
            i = 10
            return i + 15
        'Test that whether next dagrun is updated depends on run type'
        with dag_maker(dag_id='test_should_update_dag_next_dagruns_after_run_type', schedule='*/1 * * * *', max_active_runs=10) as dag:
            EmptyOperator(task_id='dummy')
        dag_model = dag_maker.dag_model
        run = dag_maker.create_dagrun(run_id='run', run_type=run_type, execution_date=DEFAULT_DATE, start_date=timezone.utcnow(), state=State.SUCCESS, session=session)
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        assert should_update is self.job_runner._should_update_dag_next_dagruns(dag, dag_model, last_dag_run=run, total_active_runs=0, session=session)

    def test_create_dag_runs(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test various invariants of _create_dag_runs.\n\n        - That the run created has the creating_job_id set\n        - That the run created is on QUEUED State\n        - That dag_model has next_dagrun\n        '
        with dag_maker(dag_id='test_create_dag_runs') as dag:
            EmptyOperator(task_id='dummy')
        dag_model = dag_maker.dag_model
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        with create_session() as session:
            self.job_runner._create_dag_runs([dag_model], session)
        dr = session.query(DagRun).filter(DagRun.dag_id == dag.dag_id).first()
        assert dr.state == State.QUEUED
        assert dr.start_date is None
        assert dag.get_last_dagrun().creating_job_id == scheduler_job.id

    @pytest.mark.need_serialized_dag
    def test_create_dag_runs_datasets(self, session, dag_maker):
        if False:
            print('Hello World!')
        '\n        Test various invariants of _create_dag_runs.\n\n        - That the run created has the creating_job_id set\n        - That the run created is on QUEUED State\n        - That dag_model has next_dagrun\n        '
        dataset1 = Dataset(uri='ds1')
        dataset2 = Dataset(uri='ds2')
        with dag_maker(dag_id='datasets-1', start_date=timezone.utcnow(), session=session):
            BashOperator(task_id='task', bash_command='echo 1', outlets=[dataset1])
        dr = dag_maker.create_dagrun(run_id='run1', execution_date=DEFAULT_DATE + timedelta(days=100), data_interval=(DEFAULT_DATE + timedelta(days=10), DEFAULT_DATE + timedelta(days=11)))
        ds1_id = session.query(DatasetModel.id).filter_by(uri=dataset1.uri).scalar()
        event1 = DatasetEvent(dataset_id=ds1_id, source_task_id='task', source_dag_id=dr.dag_id, source_run_id=dr.run_id, source_map_index=-1)
        session.add(event1)
        dr = dag_maker.create_dagrun(run_id='run2', execution_date=DEFAULT_DATE + timedelta(days=101), data_interval=(DEFAULT_DATE + timedelta(days=5), DEFAULT_DATE + timedelta(days=6)))
        event2 = DatasetEvent(dataset_id=ds1_id, source_task_id='task', source_dag_id=dr.dag_id, source_run_id=dr.run_id, source_map_index=-1)
        session.add(event2)
        with dag_maker(dag_id='datasets-consumer-multiple', schedule=[dataset1, dataset2]):
            pass
        dag2 = dag_maker.dag
        with dag_maker(dag_id='datasets-consumer-single', schedule=[dataset1]):
            pass
        dag3 = dag_maker.dag
        session = dag_maker.session
        session.add_all([DatasetDagRunQueue(dataset_id=ds1_id, target_dag_id=dag2.dag_id), DatasetDagRunQueue(dataset_id=ds1_id, target_dag_id=dag3.dag_id)])
        session.flush()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        with create_session() as session:
            self.job_runner._create_dagruns_for_dags(session, session)

        def dict_from_obj(obj):
            if False:
                while True:
                    i = 10
            'Get dict of column attrs from SqlAlchemy object.'
            return {k.key: obj.__dict__.get(k) for k in obj.__mapper__.column_attrs}
        created_run = session.query(DagRun).filter(DagRun.dag_id == dag3.dag_id).one()
        assert created_run.state == State.QUEUED
        assert created_run.start_date is None
        assert list(map(dict_from_obj, created_run.consumed_dataset_events)) == list(map(dict_from_obj, [event1, event2]))
        assert created_run.data_interval_start == DEFAULT_DATE + timedelta(days=5)
        assert created_run.data_interval_end == DEFAULT_DATE + timedelta(days=11)
        assert session.query(DatasetDagRunQueue).filter_by(target_dag_id=dag2.dag_id).one() is not None
        assert session.query(DagRun).filter(DagRun.dag_id == dag2.dag_id).one_or_none() is None
        assert session.query(DatasetDagRunQueue).filter_by(target_dag_id=dag3.dag_id).one_or_none() is None
        assert dag3.get_last_dagrun().creating_job_id == scheduler_job.id

    @time_machine.travel(DEFAULT_DATE + datetime.timedelta(days=1, seconds=9), tick=False)
    @mock.patch('airflow.jobs.scheduler_job_runner.Stats.timing')
    def test_start_dagruns(self, stats_timing, dag_maker):
        if False:
            return 10
        '\n        Test that _start_dagrun:\n\n        - moves runs to RUNNING State\n        - emit the right DagRun metrics\n        '
        with dag_maker(dag_id='test_start_dag_runs') as dag:
            EmptyOperator(task_id='dummy')
        dag_model = dag_maker.dag_model
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        with create_session() as session:
            self.job_runner._create_dag_runs([dag_model], session)
            self.job_runner._start_queued_dagruns(session)
        dr = session.query(DagRun).filter(DagRun.dag_id == dag.dag_id).first()
        assert dr.state == State.RUNNING
        stats_timing.assert_has_calls([mock.call('dagrun.schedule_delay.test_start_dag_runs', datetime.timedelta(seconds=9)), mock.call('dagrun.schedule_delay', datetime.timedelta(seconds=9), tags={'dag_id': 'test_start_dag_runs'})])
        assert dag.get_last_dagrun().creating_job_id == scheduler_job.id

    def test_extra_operator_links_not_loaded_in_scheduler_loop(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that Operator links are not loaded inside the Scheduling Loop (that does not include\n        DagFileProcessorProcess) especially the critical loop of the Scheduler.\n\n        This is to avoid running User code in the Scheduler and prevent any deadlocks\n        '
        with dag_maker(dag_id='test_extra_operator_links_not_loaded_in_scheduler') as dag:
            _ = CustomOperator(task_id='custom_task')
        custom_task = dag.task_dict['custom_task']
        assert custom_task.operator_extra_links
        session = settings.Session()
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        s_dag_2 = self.job_runner.dagbag.get_dag(dag.dag_id)
        custom_task = s_dag_2.task_dict['custom_task']
        assert not custom_task.operator_extra_links

    def test_scheduler_create_dag_runs_does_not_raise_error(self, caplog, dag_maker):
        if False:
            return 10
        '\n        Test that scheduler._create_dag_runs does not raise an error when the DAG does not exist\n        in serialized_dag table\n        '
        with dag_maker(dag_id='test_scheduler_create_dag_runs_does_not_raise_error', serialized=False):
            EmptyOperator(task_id='dummy')
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        caplog.set_level('FATAL')
        caplog.clear()
        with create_session() as session, caplog.at_level('ERROR', logger='airflow.jobs.scheduler_job_runner'):
            self.job_runner._create_dag_runs([dag_maker.dag_model], session)
            assert caplog.messages == ["DAG 'test_scheduler_create_dag_runs_does_not_raise_error' not found in serialized_dag table"]

    def test_bulk_write_to_db_external_trigger_dont_skip_scheduled_run(self, dag_maker):
        if False:
            i = 10
            return i + 15
        '\n        Test that externally triggered Dag Runs should not affect (by skipping) next\n        scheduled DAG runs\n        '
        with dag_maker(dag_id='test_bulk_write_to_db_external_trigger_dont_skip_scheduled_run', schedule='*/1 * * * *', max_active_runs=5, catchup=True) as dag:
            EmptyOperator(task_id='dummy')
        session = settings.Session()
        dag_model = dag_maker.dag_model
        assert dag_model.next_dagrun == DEFAULT_DATE
        assert dag_model.next_dagrun_data_interval_start == DEFAULT_DATE
        assert dag_model.next_dagrun_data_interval_end == DEFAULT_DATE + timedelta(minutes=1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        self.job_runner._do_scheduling(session)
        dr1 = dag.get_dagrun(DEFAULT_DATE, session=session)
        assert dr1 is not None
        assert dr1.state == State.RUNNING
        assert dr1.execution_date == DEFAULT_DATE
        assert dr1.data_interval_start == DEFAULT_DATE
        assert dr1.data_interval_end == DEFAULT_DATE + timedelta(minutes=1)
        dag_model = session.get(DagModel, dag.dag_id)
        assert dag_model.next_dagrun == DEFAULT_DATE + timedelta(minutes=1)
        assert dag_model.next_dagrun_data_interval_start == DEFAULT_DATE + timedelta(minutes=1)
        assert dag_model.next_dagrun_data_interval_end == DEFAULT_DATE + timedelta(minutes=2)
        dr = dag.create_dagrun(state=State.RUNNING, execution_date=timezone.utcnow(), run_type=DagRunType.MANUAL, session=session, external_trigger=True)
        assert dr is not None
        DAG.bulk_write_to_db([dag], session=session)
        dag_model = session.get(DagModel, dag.dag_id)
        assert dag_model.next_dagrun == DEFAULT_DATE + timedelta(minutes=1)
        assert dag_model.next_dagrun_data_interval_start == DEFAULT_DATE + timedelta(minutes=1)
        assert dag_model.next_dagrun_data_interval_end == DEFAULT_DATE + timedelta(minutes=2)

    def test_scheduler_create_dag_runs_check_existing_run(self, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Test that if a dag run exists, scheduler._create_dag_runs does not raise an error.\n        And if a Dag Run does not exist it creates next Dag Run. In both cases the Scheduler\n        sets next execution date as DagModel.next_dagrun\n        '
        with dag_maker(dag_id='test_scheduler_create_dag_runs_check_existing_run', schedule=timedelta(days=1)) as dag:
            EmptyOperator(task_id='dummy')
        session = settings.Session()
        assert dag.get_last_dagrun(session) is None
        dag_model = dag_maker.dag_model
        assert dag_model.next_dagrun == DEFAULT_DATE
        dagrun = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=dag_model.next_dagrun, start_date=timezone.utcnow(), state=State.RUNNING, external_trigger=False, session=session, creating_job_id=2)
        session.flush()
        assert dag.get_last_dagrun(session) == dagrun
        scheduler_job = Job(executor=self.null_exec)
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        self.job_runner._create_dag_runs([dag_model], session)
        assert dag_model.next_dagrun_data_interval_start == DEFAULT_DATE + timedelta(days=1)
        assert dag_model.next_dagrun_data_interval_end == DEFAULT_DATE + timedelta(days=2)
        assert dag_model.next_dagrun == DEFAULT_DATE + timedelta(days=1)
        session.rollback()

    @conf_vars({('scheduler', 'use_job_schedule'): 'false'})
    def test_do_schedule_max_active_runs_dag_timed_out(self, dag_maker):
        if False:
            return 10
        'Test that tasks are set to a finished state when their DAG times out'
        with dag_maker(dag_id='test_max_active_run_with_dag_timed_out', schedule='@once', max_active_runs=1, catchup=True, dagrun_timeout=datetime.timedelta(seconds=1)) as dag:
            task1 = BashOperator(task_id='task1', bash_command=' for((i=1;i<=600;i+=1)); do sleep "$i";  done')
        session = settings.Session()
        run1 = dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=DEFAULT_DATE, state=State.RUNNING, start_date=timezone.utcnow() - timedelta(seconds=2), session=session)
        run1_ti = run1.get_task_instance(task1.task_id, session)
        run1_ti.state = State.RUNNING
        run2 = dag.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=DEFAULT_DATE + timedelta(seconds=10), state=State.QUEUED, session=session)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        my_dag = session.get(DagModel, dag.dag_id)
        self.job_runner._create_dag_runs([my_dag], session)
        self.job_runner._schedule_dag_run(run1, session)
        run1 = session.merge(run1)
        session.refresh(run1)
        assert run1.state == State.FAILED
        assert run1_ti.state == State.SKIPPED
        session.flush()
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        run2 = session.merge(run2)
        session.refresh(run2)
        assert run2.state == State.RUNNING
        self.job_runner._schedule_dag_run(run2, session)
        run2_ti = run2.get_task_instance(task1.task_id, session)
        assert run2_ti.state == State.SCHEDULED

    def test_do_schedule_max_active_runs_task_removed(self, session, dag_maker):
        if False:
            return 10
        "Test that tasks in removed state don't count as actively running."
        with dag_maker(dag_id='test_do_schedule_max_active_runs_task_removed', start_date=DEFAULT_DATE, schedule='@once', max_active_runs=1, session=session):
            BashOperator(task_id='dummy1', bash_command='true')
        run1 = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, execution_date=DEFAULT_DATE + timedelta(hours=1), state=State.RUNNING)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        num_queued = self.job_runner._do_scheduling(session)
        assert num_queued == 1
        session.flush()
        ti = run1.task_instances[0]
        ti.refresh_from_db(session=session)
        assert ti.state == State.QUEUED

    def test_more_runs_are_not_created_when_max_active_runs_is_reached(self, dag_maker, caplog):
        if False:
            return 10
        "\n        This tests that when max_active_runs is reached, _create_dag_runs doesn't create\n        more dagruns\n        "
        with dag_maker(max_active_runs=1):
            EmptyOperator(task_id='task')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        session = settings.Session()
        assert session.query(DagRun).count() == 0
        (query, _) = DagModel.dags_needing_dagruns(session)
        dag_models = query.all()
        self.job_runner._create_dag_runs(dag_models, session)
        dr = session.query(DagRun).one()
        dr.state == DagRunState.QUEUED
        assert session.query(DagRun).count() == 1
        assert dag_maker.dag_model.next_dagrun_create_after is None
        session.flush()
        (query, _) = DagModel.dags_needing_dagruns(session)
        assert len(query.all()) == 0
        self.job_runner._create_dag_runs(dag_models, session)
        assert session.query(DagRun).count() == 1
        assert dag_maker.dag_model.next_dagrun_create_after is None
        assert dag_maker.dag_model.next_dagrun == DEFAULT_DATE
        dr = session.query(DagRun).one()
        dr.state = DagRunState.SUCCESS
        ti = dr.get_task_instance('task', session)
        ti.state = TaskInstanceState.SUCCESS
        session.merge(ti)
        session.merge(dr)
        session.flush()
        self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        (query, _) = DagModel.dags_needing_dagruns(session)
        assert len(query.all()) == 1
        assert dag_maker.dag_model.next_dagrun == DEFAULT_DATE + timedelta(days=1)
        assert session.query(DagRun).filter(DagRun.state.in_([DagRunState.RUNNING, DagRunState.QUEUED])).count() == 0

    def test_max_active_runs_creation_phasing(self, dag_maker, session):
        if False:
            i = 10
            return i + 15
        '\n        Test that when creating runs once max_active_runs is reached that the runs come in the right order\n        without gaps\n        '

        def complete_one_dagrun():
            if False:
                print('Hello World!')
            ti = session.query(TaskInstance).join(TaskInstance.dag_run).filter(TaskInstance.state != State.SUCCESS).order_by(DagRun.execution_date).first()
            if ti:
                ti.state = State.SUCCESS
                session.flush()
        self.clean_db()
        with dag_maker(max_active_runs=3, session=session) as dag:
            BashOperator(task_id='task', bash_command='true')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=True)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        (query, _) = DagModel.dags_needing_dagruns(session)
        query.all()
        for _ in range(3):
            self.job_runner._do_scheduling(session)
        model: DagModel = session.get(DagModel, dag.dag_id)
        assert DagRun.active_runs_of_dags(session=session) == {'test_dag': 3}
        assert model.next_dagrun == timezone.DateTime(2016, 1, 3, tzinfo=UTC)
        assert model.next_dagrun_create_after is None
        complete_one_dagrun()
        assert DagRun.active_runs_of_dags(session=session) == {'test_dag': 3}
        for _ in range(5):
            self.job_runner._do_scheduling(session)
            complete_one_dagrun()
        expected_execution_dates = [datetime.datetime(2016, 1, d, tzinfo=timezone.utc) for d in range(1, 6)]
        dagrun_execution_dates = [dr.execution_date for dr in session.query(DagRun).order_by(DagRun.execution_date).all()]
        assert dagrun_execution_dates == expected_execution_dates

    def test_do_schedule_max_active_runs_and_manual_trigger(self, dag_maker):
        if False:
            i = 10
            return i + 15
        "\n        Make sure that when a DAG is already at max_active_runs, that manually triggered\n        dagruns don't start running.\n        "
        with dag_maker(dag_id='test_max_active_run_plus_manual_trigger', schedule='@once', max_active_runs=1) as dag:
            task1 = BashOperator(task_id='dummy1', bash_command='true')
            task2 = BashOperator(task_id='dummy2', bash_command='true')
            task1 >> task2
            BashOperator(task_id='dummy3', bash_command='true')
        session = settings.Session()
        dag_run = dag_maker.create_dagrun(state=State.QUEUED, session=session)
        dag.sync_to_db(session=session)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        num_queued = self.job_runner._do_scheduling(session)
        dag_run = session.merge(dag_run)
        session.refresh(dag_run)
        assert num_queued == 2
        assert dag_run.state == State.RUNNING
        dag_maker.create_dagrun(run_type=DagRunType.MANUAL, execution_date=DEFAULT_DATE + timedelta(hours=1), state=State.QUEUED, session=session)
        session.flush()
        self.job_runner._do_scheduling(session)
        assert len(DagRun.find(dag_id=dag.dag_id, state=State.RUNNING, session=session)) == 1
        assert len(DagRun.find(dag_id=dag.dag_id, state=State.QUEUED, session=session)) == 1

    def test_max_active_runs_in_a_dag_doesnt_stop_running_dagruns_in_otherdags(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        session = settings.Session()
        with dag_maker('test_dag1', start_date=DEFAULT_DATE, schedule=timedelta(hours=1), max_active_runs=1):
            EmptyOperator(task_id='mytask')
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        for _ in range(29):
            dr = dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        with dag_maker('test_dag2', start_date=timezone.datetime(2020, 1, 1), schedule=timedelta(hours=1)):
            EmptyOperator(task_id='mytask')
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        for _ in range(9):
            dr = dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        dag1_running_count = session.query(func.count(DagRun.id)).filter(DagRun.dag_id == 'test_dag1', DagRun.state == State.RUNNING).scalar()
        running_count = session.query(func.count(DagRun.id)).filter(DagRun.state == State.RUNNING).scalar()
        assert dag1_running_count == 1
        assert running_count == 11

    def test_start_queued_dagruns_do_follow_execution_date_order(self, dag_maker):
        if False:
            while True:
                i = 10
        session = settings.Session()
        with dag_maker('test_dag1', max_active_runs=1) as dag:
            EmptyOperator(task_id='mytask')
        date = dag.following_schedule(DEFAULT_DATE)
        for i in range(30):
            dr = dag_maker.create_dagrun(run_id=f'dagrun_{i}', run_type=DagRunType.SCHEDULED, state=State.QUEUED, execution_date=date)
            date = dr.execution_date + timedelta(hours=1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        dr = DagRun.find(run_id='dagrun_0')
        ti = dr[0].get_task_instance(task_id='mytask', session=session)
        ti.state = State.SUCCESS
        session.merge(ti)
        session.commit()
        assert dr[0].state == State.RUNNING
        dr[0].state = State.SUCCESS
        session.merge(dr[0])
        session.flush()
        assert dr[0].state == State.SUCCESS
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        dr = DagRun.find(run_id='dagrun_1')
        assert len(session.query(DagRun).filter(DagRun.state == State.RUNNING).all()) == 1
        assert dr[0].state == State.RUNNING

    def test_no_dagruns_would_stuck_in_running(self, dag_maker):
        if False:
            return 10
        session = settings.Session()
        date = timezone.datetime(2016, 1, 1)
        with dag_maker('test_dagrun_states_are_correct_1', max_active_runs=1, start_date=date) as dag:
            task1 = EmptyOperator(task_id='dummy_task')
        dr1_running = dag_maker.create_dagrun(run_id='dr1_run_1', execution_date=date)
        dag_maker.create_dagrun(run_id='dr1_run_2', state=State.QUEUED, execution_date=dag.following_schedule(dr1_running.execution_date))
        date = timezone.datetime(2020, 1, 1)
        with dag_maker('test_dagrun_states_are_correct_2', start_date=date) as dag:
            EmptyOperator(task_id='dummy_task')
        for i in range(16):
            dr = dag_maker.create_dagrun(run_id=f'dr2_run_{i + 1}', state=State.RUNNING, execution_date=date)
            date = dr.execution_date + timedelta(hours=1)
        dr16 = DagRun.find(run_id='dr2_run_16')
        date = dr16[0].execution_date + timedelta(hours=1)
        for i in range(16, 32):
            dr = dag_maker.create_dagrun(run_id=f'dr2_run_{i + 1}', state=State.QUEUED, execution_date=date)
            date = dr.execution_date + timedelta(hours=1)
        date = timezone.datetime(2021, 1, 1)
        with dag_maker('test_dagrun_states_are_correct_3', start_date=date) as dag:
            EmptyOperator(task_id='dummy_task')
        for i in range(16):
            dr = dag_maker.create_dagrun(run_id=f'dr3_run_{i + 1}', state=State.RUNNING, execution_date=date)
            date = dr.execution_date + timedelta(hours=1)
        dr16 = DagRun.find(run_id='dr3_run_16')
        date = dr16[0].execution_date + timedelta(hours=1)
        for i in range(16, 32):
            dr = dag_maker.create_dagrun(run_id=f'dr2_run_{i + 1}', state=State.QUEUED, execution_date=date)
            date = dr.execution_date + timedelta(hours=1)
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor(do_update=False)
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        ti = TaskInstance(task=task1, execution_date=DEFAULT_DATE)
        ti.refresh_from_db()
        ti.state = State.SUCCESS
        session.merge(ti)
        session.flush()
        with mock.patch.object(settings, 'USE_JOB_SCHEDULE', False):
            self.job_runner._do_scheduling(session)
            self.job_runner._do_scheduling(session)
        assert DagRun.find(run_id='dr1_run_1')[0].state == State.SUCCESS
        assert DagRun.find(run_id='dr1_run_2')[0].state == State.RUNNING

    @pytest.mark.parametrize('state, start_date, end_date', [[State.NONE, None, None], [State.UP_FOR_RETRY, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)], [State.UP_FOR_RESCHEDULE, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)]])
    def test_dag_file_processor_process_task_instances(self, state, start_date, end_date, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if _process_task_instances puts the right task instances into the\n        mock_list.\n        '
        with dag_maker(dag_id='test_scheduler_process_execute_task'):
            BashOperator(task_id='dummy', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        assert dr is not None
        with create_session() as session:
            ti = dr.get_task_instances(session=session)[0]
            ti.state = state
            ti.start_date = start_date
            ti.end_date = end_date
            self.job_runner._schedule_dag_run(dr, session)
            assert session.query(TaskInstance).filter_by(state=State.SCHEDULED).count() == 1
            session.refresh(ti)
            assert ti.state == State.SCHEDULED

    @pytest.mark.parametrize('state,start_date,end_date', [[State.NONE, None, None], [State.UP_FOR_RETRY, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)], [State.UP_FOR_RESCHEDULE, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)]])
    def test_dag_file_processor_process_task_instances_with_max_active_tis_per_dag(self, state, start_date, end_date, dag_maker):
        if False:
            print('Hello World!')
        '\n        Test if _process_task_instances puts the right task instances into the\n        mock_list.\n        '
        with dag_maker(dag_id='test_scheduler_process_execute_task_with_max_active_tis_per_dag'):
            BashOperator(task_id='dummy', max_active_tis_per_dag=2, bash_command='echo Hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        assert dr is not None
        with create_session() as session:
            ti = dr.get_task_instances(session=session)[0]
            ti.state = state
            ti.start_date = start_date
            ti.end_date = end_date
            self.job_runner._schedule_dag_run(dr, session)
            assert session.query(TaskInstance).filter_by(state=State.SCHEDULED).count() == 1
            session.refresh(ti)
            assert ti.state == State.SCHEDULED

    @pytest.mark.parametrize('state,start_date,end_date', [[State.NONE, None, None], [State.UP_FOR_RETRY, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)], [State.UP_FOR_RESCHEDULE, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)]])
    def test_dag_file_processor_process_task_instances_with_max_active_tis_per_dagrun(self, state, start_date, end_date, dag_maker):
        if False:
            return 10
        '\n        Test if _process_task_instances puts the right task instances into the\n        mock_list.\n        '
        with dag_maker(dag_id='test_scheduler_process_execute_task_with_max_active_tis_per_dagrun'):
            BashOperator(task_id='dummy', max_active_tis_per_dagrun=2, bash_command='echo Hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        assert dr is not None
        with create_session() as session:
            ti = dr.get_task_instances(session=session)[0]
            ti.state = state
            ti.start_date = start_date
            ti.end_date = end_date
            self.job_runner._schedule_dag_run(dr, session)
            assert session.query(TaskInstance).filter_by(state=State.SCHEDULED).count() == 1
            session.refresh(ti)
            assert ti.state == State.SCHEDULED

    @pytest.mark.parametrize('state, start_date, end_date', [[State.NONE, None, None], [State.UP_FOR_RETRY, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)], [State.UP_FOR_RESCHEDULE, timezone.utcnow() - datetime.timedelta(minutes=30), timezone.utcnow() - datetime.timedelta(minutes=15)]])
    def test_dag_file_processor_process_task_instances_depends_on_past(self, state, start_date, end_date, dag_maker):
        if False:
            print('Hello World!')
        '\n        Test if _process_task_instances puts the right task instances into the\n        mock_list.\n        '
        with dag_maker(dag_id='test_scheduler_process_execute_task_depends_on_past', default_args={'depends_on_past': True}):
            BashOperator(task_id='dummy1', bash_command='echo hi')
            BashOperator(task_id='dummy2', bash_command='echo hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        assert dr is not None
        with create_session() as session:
            tis = dr.get_task_instances(session=session)
            for ti in tis:
                ti.state = state
                ti.start_date = start_date
                ti.end_date = end_date
            self.job_runner._schedule_dag_run(dr, session)
            assert session.query(TaskInstance).filter_by(state=State.SCHEDULED).count() == 2
            session.refresh(tis[0])
            session.refresh(tis[1])
            assert tis[0].state == State.SCHEDULED
            assert tis[1].state == State.SCHEDULED

    def test_scheduler_job_add_new_task(self, dag_maker):
        if False:
            return 10
        '\n        Test if a task instance will be added if the dag is updated\n        '
        with dag_maker(dag_id='test_scheduler_add_new_task') as dag:
            BashOperator(task_id='dummy', bash_command='echo test')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.dagbag = dag_maker.dagbag
        session = settings.Session()
        orm_dag = dag_maker.dag_model
        assert orm_dag is not None
        if self.job_runner.processor_agent:
            self.job_runner.processor_agent.end()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dag = self.job_runner.dagbag.get_dag('test_scheduler_add_new_task', session=session)
        self.job_runner._create_dag_runs([orm_dag], session)
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        tis = dr.get_task_instances()
        assert len(tis) == 1
        BashOperator(task_id='dummy2', dag=dag, bash_command='echo test')
        SerializedDagModel.write_dag(dag=dag)
        self.job_runner._schedule_dag_run(dr, session)
        assert session.query(TaskInstance).filter_by(state=State.SCHEDULED).count() == 2
        session.flush()
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        tis = dr.get_task_instances()
        assert len(tis) == 2

    def test_runs_respected_after_clear(self, dag_maker):
        if False:
            return 10
        '\n        Test dag after dag.clear, max_active_runs is respected\n        '
        with dag_maker(dag_id='test_scheduler_max_active_runs_respected_after_clear', start_date=DEFAULT_DATE, max_active_runs=1) as dag:
            BashOperator(task_id='dummy', bash_command='echo Hi')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        session = settings.Session()
        dr = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        dr = dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        dag_maker.create_dagrun_after(dr, run_type=DagRunType.SCHEDULED, state=State.QUEUED)
        dag.clear()
        assert len(DagRun.find(dag_id=dag.dag_id, state=State.QUEUED, session=session)) == 3
        session = settings.Session()
        self.job_runner._start_queued_dagruns(session)
        session.flush()
        assert len(DagRun.find(dag_id=dag.dag_id, state=State.RUNNING, session=session)) == 1
        assert len(DagRun.find(dag_id=dag.dag_id, state=State.QUEUED, session=session)) == 2

    def test_timeout_triggers(self, dag_maker):
        if False:
            while True:
                i = 10
        '\n        Tests that tasks in the deferred state, but whose trigger timeout\n        has expired, are correctly failed.\n\n        '
        session = settings.Session()
        with dag_maker(dag_id='test_timeout_triggers', start_date=DEFAULT_DATE, schedule='@once', max_active_runs=1, session=session):
            EmptyOperator(task_id='dummy1')
        dr1 = dag_maker.create_dagrun()
        dr2 = dag_maker.create_dagrun(run_id='test2', execution_date=DEFAULT_DATE + datetime.timedelta(seconds=1))
        ti1 = dr1.get_task_instance('dummy1', session)
        ti2 = dr2.get_task_instance('dummy1', session)
        ti1.state = State.DEFERRED
        ti1.trigger_timeout = timezone.utcnow() - datetime.timedelta(seconds=60)
        ti2.state = State.DEFERRED
        ti2.trigger_timeout = timezone.utcnow() + datetime.timedelta(seconds=60)
        session.flush()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.check_trigger_timeouts(session=session)
        session.refresh(ti1)
        session.refresh(ti2)
        assert ti1.state == State.SCHEDULED
        assert ti1.next_method == '__fail__'
        assert ti2.state == State.DEFERRED

    def test_find_zombies_nothing(self):
        if False:
            print('Hello World!')
        executor = MockExecutor(do_update=False)
        scheduler_job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(scheduler_job)
        self.job_runner.processor_agent = mock.MagicMock()
        self.job_runner._find_zombies()
        scheduler_job.executor.callback_sink.send.assert_not_called()

    def test_find_zombies(self, load_examples):
        if False:
            print('Hello World!')
        dagbag = DagBag(TEST_DAG_FOLDER, read_dags_from_db=False)
        with create_session() as session:
            session.query(Job).delete()
            dag = dagbag.get_dag('example_branch_operator')
            dag.sync_to_db()
            dag_run = dag.create_dagrun(state=DagRunState.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            scheduler_job.executor = MockExecutor()
            self.job_runner.processor_agent = mock.MagicMock()
            tasks_to_setup = ['branching', 'run_this_first']
            for task_id in tasks_to_setup:
                task = dag.get_task(task_id=task_id)
                ti = TaskInstance(task, run_id=dag_run.run_id, state=State.RUNNING)
                ti.queued_by_job_id = 999
                local_job = Job(dag_id=ti.dag_id)
                LocalTaskJobRunner(job=local_job, task_instance=ti)
                local_job.state = TaskInstanceState.FAILED
                session.add(local_job)
                session.flush()
                ti.job_id = local_job.id
                session.add(ti)
                session.flush()
            assert task.task_id == 'run_this_first'
            ti.queued_by_job_id = scheduler_job.id
            session.flush()
            self.job_runner._find_zombies()
        scheduler_job.executor.callback_sink.send.assert_called_once()
        requests = scheduler_job.executor.callback_sink.send.call_args.args
        assert 1 == len(requests)
        assert requests[0].full_filepath == dag.fileloc
        assert requests[0].msg == str(self.job_runner._generate_zombie_message_details(ti))
        assert requests[0].is_failure_callback is True
        assert isinstance(requests[0].simple_task_instance, SimpleTaskInstance)
        assert ti.dag_id == requests[0].simple_task_instance.dag_id
        assert ti.task_id == requests[0].simple_task_instance.task_id
        assert ti.run_id == requests[0].simple_task_instance.run_id
        assert ti.map_index == requests[0].simple_task_instance.map_index
        with create_session() as session:
            session.query(TaskInstance).delete()
            session.query(Job).delete()

    def test_zombie_message(self, load_examples):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the zombie message comes out as expected\n        '
        dagbag = DagBag(TEST_DAG_FOLDER, read_dags_from_db=False)
        with create_session() as session:
            session.query(Job).delete()
            dag = dagbag.get_dag('example_branch_operator')
            dag.sync_to_db()
            dag_run = dag.create_dagrun(state=DagRunState.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            scheduler_job = Job(executor=MockExecutor())
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            self.job_runner.processor_agent = mock.MagicMock()
            tasks_to_setup = ['branching', 'run_this_first']
            for task_id in tasks_to_setup:
                task = dag.get_task(task_id=task_id)
                ti = TaskInstance(task, run_id=dag_run.run_id, state=State.RUNNING)
                ti.queued_by_job_id = 999
                local_job = Job(dag_id=ti.dag_id)
                local_job.state = TaskInstanceState.FAILED
                session.add(local_job)
                session.flush()
                ti.job_id = local_job.id
                session.add(ti)
                session.flush()
            assert task.task_id == 'run_this_first'
            ti.queued_by_job_id = scheduler_job.id
            session.flush()
            zombie_message = self.job_runner._generate_zombie_message_details(ti)
            assert zombie_message == {'DAG Id': 'example_branch_operator', 'Task Id': 'run_this_first', 'Run Id': 'scheduled__2016-01-01T00:00:00+00:00'}
            ti.hostname = '10.10.10.10'
            ti.map_index = 2
            ti.external_executor_id = 'abcdefg'
            zombie_message = self.job_runner._generate_zombie_message_details(ti)
            assert zombie_message == {'DAG Id': 'example_branch_operator', 'Task Id': 'run_this_first', 'Run Id': 'scheduled__2016-01-01T00:00:00+00:00', 'Hostname': '10.10.10.10', 'Map Index': 2, 'External Executor Id': 'abcdefg'}

    def test_find_zombies_handle_failure_callbacks_are_correctly_passed_to_dag_processor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the same set of failure callback with zombies are passed to the dag\n        file processors until the next zombie detection logic is invoked.\n        '
        with conf_vars({('core', 'load_examples'): 'False'}), create_session() as session:
            dagbag = DagBag(dag_folder=os.path.join(settings.DAGS_FOLDER, 'test_example_bash_operator.py'), read_dags_from_db=False)
            session.query(Job).delete()
            dag = dagbag.get_dag('test_example_bash_operator')
            dag.sync_to_db(processor_subdir=TEST_DAG_FOLDER)
            dag_run = dag.create_dagrun(state=DagRunState.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            task = dag.get_task(task_id='run_this_last')
            ti = TaskInstance(task, run_id=dag_run.run_id, state=State.RUNNING)
            local_job = Job(dag_id=ti.dag_id)
            LocalTaskJobRunner(job=local_job, task_instance=ti)
            local_job.state = JobState.FAILED
            session.add(local_job)
            session.flush()
            session.add(ti)
            ti.job_id = local_job.id
            session.flush()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.processor_agent = mock.MagicMock()
        self.job_runner._find_zombies()
        scheduler_job.executor.callback_sink.send.assert_called_once()
        expected_failure_callback_requests = [TaskCallbackRequest(full_filepath=dag.fileloc, simple_task_instance=SimpleTaskInstance.from_ti(ti), processor_subdir=TEST_DAG_FOLDER, msg=str(self.job_runner._generate_zombie_message_details(ti)))]
        callback_requests = scheduler_job.executor.callback_sink.send.call_args.args
        assert len(callback_requests) == 1
        assert {zombie.simple_task_instance.key for zombie in expected_failure_callback_requests} == {result.simple_task_instance.key for result in callback_requests}
        expected_failure_callback_requests[0].simple_task_instance = None
        callback_requests[0].simple_task_instance = None
        assert expected_failure_callback_requests[0] == callback_requests[0]

    def test_cleanup_stale_dags(self):
        if False:
            for i in range(10):
                print('nop')
        dagbag = DagBag(TEST_DAG_FOLDER, read_dags_from_db=False)
        with create_session() as session:
            dag = dagbag.get_dag('test_example_bash_operator')
            dag.sync_to_db()
            dm = DagModel.get_current('test_example_bash_operator')
            dm.last_parsed_time = timezone.utcnow() - timedelta(minutes=11)
            session.merge(dm)
            dag = dagbag.get_dag('test_start_date_scheduling')
            dag.sync_to_db()
            session.flush()
            scheduler_job = Job(executor=MockExecutor())
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
            self.job_runner.processor_agent = mock.MagicMock()
            active_dag_count = session.query(func.count(DagModel.dag_id)).filter(DagModel.is_active).scalar()
            assert active_dag_count == 2
            self.job_runner._cleanup_stale_dags(session)
            session.flush()
            active_dag_count = session.query(func.count(DagModel.dag_id)).filter(DagModel.is_active).scalar()
            assert active_dag_count == 1

    @mock.patch.object(settings, 'USE_JOB_SCHEDULE', False)
    def run_scheduler_until_dagrun_terminal(self, job_runner: SchedulerJobRunner):
        if False:
            return 10
        '\n        Run a scheduler until any dag run reaches a terminal state, or the scheduler becomes "idle".\n\n        This needs a DagRun to be pre-created (it can be in running or queued state) as no more will be\n        created as we turn off creating new DagRuns via setting USE_JOB_SCHEDULE to false\n\n        Note: This doesn\'t currently account for tasks that go into retry -- the scheduler would be detected\n        as idle in that circumstance\n        '

        def spy_on_return(orig, result):
            if False:
                print('Hello World!')

            def spy(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                ret = orig(*args, **kwargs)
                result.append(ret)
                return ret
            return spy
        num_queued_tis: deque[int] = deque([], 3)
        num_finished_events: deque[int] = deque([], 3)
        do_scheduling_spy = mock.patch.object(job_runner, '_do_scheduling', side_effect=spy_on_return(job_runner._do_scheduling, num_queued_tis))
        executor_events_spy = mock.patch.object(job_runner, '_process_executor_events', side_effect=spy_on_return(job_runner._process_executor_events, num_finished_events))
        orig_set_state = DagRun.set_state

        def watch_set_state(self: DagRun, state, **kwargs):
            if False:
                return 10
            if state in (DagRunState.SUCCESS, DagRunState.FAILED):
                job_runner.num_runs = 1
            orig_set_state(self, state, **kwargs)

        def watch_heartbeat(*args, **kwargs):
            if False:
                print('Hello World!')
            if len(num_queued_tis) < 3 or len(num_finished_events) < 3:
                return
            queued_any_tis = any((val > 0 for val in num_queued_tis))
            finished_any_events = any((val > 0 for val in num_finished_events))
            assert queued_any_tis or finished_any_events, 'Scheduler has stalled without setting the DagRun state!'
        set_state_spy = mock.patch.object(DagRun, 'set_state', new=watch_set_state)
        heartbeat_spy = mock.patch.object(job_runner, 'heartbeat', new=watch_heartbeat)
        with heartbeat_spy, set_state_spy, do_scheduling_spy, executor_events_spy:
            run_job(job_runner.job, execute_callable=job_runner._execute)

    @pytest.mark.long_running
    @pytest.mark.parametrize('dag_id', ['test_mapped_classic', 'test_mapped_taskflow'])
    def test_mapped_dag(self, dag_id, session):
        if False:
            print('Hello World!')
        'End-to-end test of a simple mapped dag'
        from airflow.executors.sequential_executor import SequentialExecutor
        self.dagbag.process_file(str(TEST_DAGS_FOLDER / f'{dag_id}.py'))
        dag = self.dagbag.get_dag(dag_id)
        assert dag
        dr = dag.create_dagrun(run_type=DagRunType.MANUAL, start_date=timezone.utcnow(), state=State.RUNNING, execution_date=timezone.utcnow() - datetime.timedelta(days=2), session=session)
        executor = SequentialExecutor()
        job = Job(executor=executor)
        self.job_runner = SchedulerJobRunner(job=job, subdir=dag.fileloc)
        self.run_scheduler_until_dagrun_terminal(job)
        dr.refresh_from_db(session)
        assert dr.state == DagRunState.SUCCESS

    def test_should_mark_empty_task_as_success(self):
        if False:
            while True:
                i = 10
        dag_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dags/test_only_empty_tasks.py')
        dagbag = DagBag(dag_folder=dag_file, include_examples=False, read_dags_from_db=False)
        dagbag.sync_to_db()
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner.processor_agent = mock.MagicMock()
        dag = self.job_runner.dagbag.get_dag('test_only_empty_tasks')
        session = settings.Session()
        orm_dag = session.get(DagModel, dag.dag_id)
        self.job_runner._create_dag_runs([orm_dag], session)
        drs = DagRun.find(dag_id=dag.dag_id, session=session)
        assert len(drs) == 1
        dr = drs[0]
        self.job_runner._schedule_dag_run(dr, session)
        with create_session() as session:
            tis = session.query(TaskInstance).all()
        dags = self.job_runner.dagbag.dags.values()
        assert ['test_only_empty_tasks'] == [dag.dag_id for dag in dags]
        assert 6 == len(tis)
        assert {('test_task_a', 'success'), ('test_task_b', None), ('test_task_c', 'success'), ('test_task_on_execute', 'scheduled'), ('test_task_on_success', 'scheduled'), ('test_task_outlets', 'scheduled')} == {(ti.task_id, ti.state) for ti in tis}
        for (state, start_date, end_date, duration) in [(ti.state, ti.start_date, ti.end_date, ti.duration) for ti in tis]:
            if state == 'success':
                assert start_date is not None
                assert end_date is not None
                assert 0.0 == duration
            else:
                assert start_date is None
                assert end_date is None
                assert duration is None
        self.job_runner._schedule_dag_run(dr, session)
        with create_session() as session:
            tis = session.query(TaskInstance).all()
        assert 6 == len(tis)
        assert {('test_task_a', 'success'), ('test_task_b', 'success'), ('test_task_c', 'success'), ('test_task_on_execute', 'scheduled'), ('test_task_on_success', 'scheduled'), ('test_task_outlets', 'scheduled')} == {(ti.task_id, ti.state) for ti in tis}
        for (state, start_date, end_date, duration) in [(ti.state, ti.start_date, ti.end_date, ti.duration) for ti in tis]:
            if state == 'success':
                assert start_date is not None
                assert end_date is not None
                assert 0.0 == duration
            else:
                assert start_date is None
                assert end_date is None
                assert duration is None

    @pytest.mark.need_serialized_dag
    def test_catchup_works_correctly(self, dag_maker):
        if False:
            i = 10
            return i + 15
        'Test that catchup works correctly'
        session = settings.Session()
        with dag_maker(dag_id='test_catchup_schedule_dag', schedule=timedelta(days=1), start_date=DEFAULT_DATE, catchup=True, max_active_runs=1, session=session) as dag:
            EmptyOperator(task_id='dummy')
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner.processor_agent = mock.MagicMock(spec=DagFileProcessorAgent)
        self.job_runner._create_dag_runs([dag_maker.dag_model], session)
        self.job_runner._start_queued_dagruns(session)
        dr = DagRun.find(execution_date=DEFAULT_DATE, session=session)[0]
        ti = dr.get_task_instance(task_id='dummy')
        ti.state = State.SUCCESS
        session.merge(ti)
        session.flush()
        self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        self.job_runner._schedule_dag_run(dr, session)
        session.flush()
        dag.catchup = False
        dag.sync_to_db()
        assert not dag.catchup
        dm = DagModel.get_dagmodel(dag.dag_id)
        self.job_runner._create_dag_runs([dm], session)
        assert session.query(DagRun.execution_date).filter(DagRun.execution_date != DEFAULT_DATE).scalar() > timezone.utcnow() - timedelta(days=2)

    def test_update_dagrun_state_for_paused_dag(self, dag_maker, session):
        if False:
            while True:
                i = 10
        'Test that _update_dagrun_state_for_paused_dag puts DagRuns in terminal states'
        with dag_maker('testdag') as dag:
            EmptyOperator(task_id='task1')
        scheduled_run = dag_maker.create_dagrun(execution_date=datetime.datetime(2022, 1, 1), run_type=DagRunType.SCHEDULED)
        scheduled_run.last_scheduling_decision = datetime.datetime.now(timezone.utc) - timedelta(minutes=1)
        ti = scheduled_run.get_task_instances()[0]
        ti.set_state(TaskInstanceState.RUNNING)
        dm = DagModel.get_dagmodel(dag.dag_id)
        dm.is_paused = True
        session.merge(dm)
        session.merge(ti)
        session.flush()
        assert scheduled_run.state == State.RUNNING
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner._update_dag_run_state_for_paused_dags(session=session)
        session.flush()
        (scheduled_run,) = DagRun.find(dag_id=dag.dag_id, run_type=DagRunType.SCHEDULED, session=session)
        assert scheduled_run.state == State.RUNNING
        prior_last_scheduling_decision = scheduled_run.last_scheduling_decision
        self.job_runner._update_dag_run_state_for_paused_dags(session=session)
        (scheduled_run,) = DagRun.find(dag_id=dag.dag_id, run_type=DagRunType.SCHEDULED, session=session)
        assert scheduled_run.state == State.RUNNING
        assert prior_last_scheduling_decision == scheduled_run.last_scheduling_decision
        ti.set_state(TaskInstanceState.SUCCESS)
        self.job_runner._update_dag_run_state_for_paused_dags(session=session)
        (scheduled_run,) = DagRun.find(dag_id=dag.dag_id, run_type=DagRunType.SCHEDULED, session=session)
        assert scheduled_run.state == State.SUCCESS

    def test_update_dagrun_state_for_paused_dag_not_for_backfill(self, dag_maker, session):
        if False:
            while True:
                i = 10
        'Test that the _update_dagrun_state_for_paused_dag does not affect backfilled dagruns'
        with dag_maker('testdag') as dag:
            EmptyOperator(task_id='task1')
        backfill_run = dag_maker.create_dagrun(run_type=DagRunType.BACKFILL_JOB)
        backfill_run.last_scheduling_decision = datetime.datetime.now(timezone.utc) - timedelta(minutes=1)
        ti = backfill_run.get_task_instances()[0]
        ti.set_state(TaskInstanceState.SUCCESS)
        dm = DagModel.get_dagmodel(dag.dag_id)
        dm.is_paused = True
        session.merge(dm)
        session.merge(ti)
        session.flush()
        assert backfill_run.state == State.RUNNING
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        scheduler_job.executor = MockExecutor()
        self.job_runner._update_dag_run_state_for_paused_dags()
        session.flush()
        (backfill_run,) = DagRun.find(dag_id=dag.dag_id, run_type=DagRunType.BACKFILL_JOB, session=session)
        assert backfill_run.state == State.RUNNING

    def test_dataset_orphaning(self, dag_maker, session):
        if False:
            print('Hello World!')
        dataset1 = Dataset(uri='ds1')
        dataset2 = Dataset(uri='ds2')
        dataset3 = Dataset(uri='ds3')
        dataset4 = Dataset(uri='ds4')
        with dag_maker(dag_id='datasets-1', schedule=[dataset1, dataset2], session=session):
            BashOperator(task_id='task', bash_command='echo 1', outlets=[dataset3, dataset4])
        non_orphaned_dataset_count = session.query(DatasetModel).filter(~DatasetModel.is_orphaned).count()
        assert non_orphaned_dataset_count == 4
        orphaned_dataset_count = session.query(DatasetModel).filter(DatasetModel.is_orphaned).count()
        assert orphaned_dataset_count == 0
        with dag_maker(dag_id='datasets-1', schedule=[dataset1], session=session):
            BashOperator(task_id='task', bash_command='echo 1', outlets=[dataset3])
        scheduler_job = Job()
        self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        self.job_runner._orphan_unreferenced_datasets(session=session)
        session.flush()
        non_orphaned_datasets = [dataset.uri for dataset in session.query(DatasetModel.uri).filter(~DatasetModel.is_orphaned).order_by(DatasetModel.uri)]
        assert non_orphaned_datasets == ['ds1', 'ds3']
        orphaned_datasets = [dataset.uri for dataset in session.query(DatasetModel.uri).filter(DatasetModel.is_orphaned).order_by(DatasetModel.uri)]
        assert orphaned_datasets == ['ds2', 'ds4']

    def test_misconfigured_dags_doesnt_crash_scheduler(self, session, dag_maker, caplog):
        if False:
            return 10
        "Test that if dagrun creation throws an exception, the scheduler doesn't crash"
        with dag_maker('testdag1', serialized=True):
            BashOperator(task_id='task', bash_command='echo 1')
        dm1 = dag_maker.dag_model
        dm1.next_dagrun = None
        session.add(dm1)
        session.flush()
        with dag_maker('testdag2', serialized=True):
            BashOperator(task_id='task', bash_command='echo 1')
        dm2 = dag_maker.dag_model
        scheduler_job = Job()
        job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
        job_runner._create_dag_runs([dm1, dm2], session)
        assert 'Failed creating DagRun for testdag1' in caplog.text
        assert not DagRun.find(dag_id='testdag1', session=session)
        assert DagRun.find(dag_id='testdag2', session=session)

@pytest.mark.need_serialized_dag
def test_schedule_dag_run_with_upstream_skip(dag_maker, session):
    if False:
        print('Hello World!')
    '\n    Test if _schedule_dag_run puts a task instance into SKIPPED state if any of its\n    upstream tasks are skipped according to TriggerRuleDep.\n    '
    with dag_maker(dag_id='test_task_with_upstream_skip_process_task_instances', start_date=DEFAULT_DATE, session=session):
        dummy1 = EmptyOperator(task_id='dummy1')
        dummy2 = EmptyOperator(task_id='dummy2')
        dummy3 = EmptyOperator(task_id='dummy3')
        [dummy1, dummy2] >> dummy3
    dr = dag_maker.create_dagrun(state=State.RUNNING)
    assert dr is not None
    tis = {ti.task_id: ti for ti in dr.get_task_instances(session=session)}
    tis[dummy1.task_id].state = State.SKIPPED
    tis[dummy2.task_id].state = State.SUCCESS
    assert tis[dummy3.task_id].state == State.NONE
    session.flush()
    scheduler_job = Job()
    job_runner = SchedulerJobRunner(job=scheduler_job, subdir=os.devnull)
    job_runner._schedule_dag_run(dr, session)
    session.flush()
    tis = {ti.task_id: ti for ti in dr.get_task_instances(session=session)}
    assert tis[dummy1.task_id].state == State.SKIPPED
    assert tis[dummy2.task_id].state == State.SUCCESS
    assert tis[dummy3.task_id].state == State.SKIPPED

class TestSchedulerJobQueriesCount:
    """
    These tests are designed to detect changes in the number of queries for
    different DAG files. These tests allow easy detection when a change is
    made that affects the performance of the SchedulerJob.
    """
    scheduler_job: Job | None

    @staticmethod
    def clean_db():
        if False:
            for i in range(10):
                print('nop')
        clear_db_runs()
        clear_db_pools()
        clear_db_dags()
        clear_db_sla_miss()
        clear_db_import_errors()
        clear_db_jobs()
        clear_db_serialized_dags()

    @pytest.fixture(autouse=True)
    def per_test(self) -> Generator:
        if False:
            print('Hello World!')
        self.clean_db()
        yield
        if self.job_runner.processor_agent:
            self.job_runner.processor_agent.end()
        self.clean_db()

    @pytest.mark.parametrize('expected_query_count, dag_count, task_count', [(21, 1, 1), (21, 1, 5), (93, 10, 10)])
    def test_execute_queries_count_with_harvested_dags(self, expected_query_count, dag_count, task_count):
        if False:
            return 10
        with mock.patch.dict('os.environ', {'PERF_DAGS_COUNT': str(dag_count), 'PERF_TASKS_COUNT': str(task_count), 'PERF_START_AGO': '1d', 'PERF_SCHEDULE_INTERVAL': '30m', 'PERF_SHAPE': 'no_structure'}), conf_vars({('scheduler', 'use_job_schedule'): 'True', ('core', 'load_examples'): 'False', ('core', 'min_serialized_dag_update_interval'): '100', ('core', 'min_serialized_dag_fetch_interval'): '100'}):
            dagruns = []
            dagbag = DagBag(dag_folder=ELASTIC_DAG_FILE, include_examples=False, read_dags_from_db=False)
            dagbag.sync_to_db()
            dag_ids = dagbag.dag_ids
            dagbag = DagBag(read_dags_from_db=True)
            for (i, dag_id) in enumerate(dag_ids):
                dag = dagbag.get_dag(dag_id)
                dr = dag.create_dagrun(state=State.RUNNING, run_id=f'{DagRunType.MANUAL.value}__{i}', dag_hash=dagbag.dags_hash[dag.dag_id])
                dagruns.append(dr)
                for ti in dr.get_task_instances():
                    ti.set_state(state=State.SCHEDULED)
            mock_agent = mock.MagicMock()
            scheduler_job = Job(executor=MockExecutor(do_update=False))
            scheduler_job.heartbeat = mock.MagicMock()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=PERF_DAGS_FOLDER, num_runs=1)
            self.job_runner.processor_agent = mock_agent
            with assert_queries_count(expected_query_count, margin=15):
                with mock.patch.object(DagRun, 'next_dagruns_to_examine') as mock_dagruns:
                    query = MagicMock()
                    query.all.return_value = dagruns
                    mock_dagruns.return_value = query
                    self.job_runner._run_scheduler_loop()

    @pytest.mark.parametrize('expected_query_counts, dag_count, task_count, start_ago, schedule_interval, shape', [([10, 10, 10, 10], 1, 1, '1d', 'None', 'no_structure'), ([10, 10, 10, 10], 1, 1, '1d', 'None', 'linear'), ([24, 14, 14, 14], 1, 1, '1d', '@once', 'no_structure'), ([24, 14, 14, 14], 1, 1, '1d', '@once', 'linear'), ([24, 26, 29, 32], 1, 1, '1d', '30m', 'no_structure'), ([24, 26, 29, 32], 1, 1, '1d', '30m', 'linear'), ([24, 26, 29, 32], 1, 1, '1d', '30m', 'binary_tree'), ([24, 26, 29, 32], 1, 1, '1d', '30m', 'star'), ([24, 26, 29, 32], 1, 1, '1d', '30m', 'grid'), ([10, 10, 10, 10], 1, 5, '1d', 'None', 'no_structure'), ([10, 10, 10, 10], 1, 5, '1d', 'None', 'linear'), ([24, 14, 14, 14], 1, 5, '1d', '@once', 'no_structure'), ([25, 15, 15, 15], 1, 5, '1d', '@once', 'linear'), ([24, 26, 29, 32], 1, 5, '1d', '30m', 'no_structure'), ([25, 28, 32, 36], 1, 5, '1d', '30m', 'linear'), ([25, 28, 32, 36], 1, 5, '1d', '30m', 'binary_tree'), ([25, 28, 32, 36], 1, 5, '1d', '30m', 'star'), ([25, 28, 32, 36], 1, 5, '1d', '30m', 'grid'), ([10, 10, 10, 10], 10, 10, '1d', 'None', 'no_structure'), ([10, 10, 10, 10], 10, 10, '1d', 'None', 'linear'), ([105, 38, 38, 38], 10, 10, '1d', '@once', 'no_structure'), ([115, 51, 51, 51], 10, 10, '1d', '@once', 'linear'), ([105, 119, 119, 119], 10, 10, '1d', '30m', 'no_structure'), ([115, 145, 145, 145], 10, 10, '1d', '30m', 'linear'), ([115, 139, 139, 139], 10, 10, '1d', '30m', 'binary_tree'), ([115, 139, 139, 139], 10, 10, '1d', '30m', 'star'), ([115, 139, 139, 139], 10, 10, '1d', '30m', 'grid')])
    def test_process_dags_queries_count(self, expected_query_counts, dag_count, task_count, start_ago, schedule_interval, shape):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.dict('os.environ', {'PERF_DAGS_COUNT': str(dag_count), 'PERF_TASKS_COUNT': str(task_count), 'PERF_START_AGO': start_ago, 'PERF_SCHEDULE_INTERVAL': schedule_interval, 'PERF_SHAPE': shape}), conf_vars({('scheduler', 'use_job_schedule'): 'True', ('core', 'min_serialized_dag_update_interval'): '100', ('core', 'min_serialized_dag_fetch_interval'): '100'}):
            dagbag = DagBag(dag_folder=ELASTIC_DAG_FILE, include_examples=False)
            dagbag.sync_to_db()
            mock_agent = mock.MagicMock()
            scheduler_job = Job(job_type=SchedulerJobRunner.job_type, executor=MockExecutor(do_update=False))
            scheduler_job.heartbeat = mock.MagicMock()
            self.job_runner = SchedulerJobRunner(job=scheduler_job, subdir=PERF_DAGS_FOLDER, num_runs=1)
            self.job_runner.processor_agent = mock_agent
            failures = []
            message = 'Expected {expected_count} query, but got {current_count} located at:'
            for expected_query_count in expected_query_counts:
                with create_session() as session:
                    try:
                        with assert_queries_count(expected_query_count, message_fmt=message, margin=15):
                            self.job_runner._do_scheduling(session)
                    except AssertionError as e:
                        failures.append(str(e))
            if failures:
                prefix = 'Collected database query count mismatches:'
                joined = '\n\n'.join(failures)
                raise AssertionError(f'{prefix}\n\n{joined}')