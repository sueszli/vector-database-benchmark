from __future__ import annotations
import datetime
import pytest
import time_machine
from airflow import settings
from airflow.models import DagRun, TaskInstance
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.utils import timezone
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.types import DagRunType
from tests.test_utils.db import clear_db_runs, clear_db_xcom
pytestmark = pytest.mark.db_test
DEFAULT_DATE = timezone.datetime(2016, 1, 1)
END_DATE = timezone.datetime(2016, 1, 2)
INTERVAL = datetime.timedelta(hours=12)
FROZEN_NOW = timezone.datetime(2016, 1, 2, 12, 1, 1)

def get_task_instances(task_id):
    if False:
        return 10
    session = settings.Session()
    return session.query(TaskInstance).join(TaskInstance.dag_run).filter(TaskInstance.task_id == task_id).order_by(DagRun.execution_date).all()

class TestLatestOnlyOperator:

    @staticmethod
    def clean_db():
        if False:
            print('Hello World!')
        clear_db_runs()
        clear_db_xcom()

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.clean_db()

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.dag = DAG('test_dag', default_args={'owner': 'airflow', 'start_date': DEFAULT_DATE}, schedule=INTERVAL)
        self.freezer = time_machine.travel(FROZEN_NOW, tick=False)
        self.freezer.start()

    def teardown_method(self):
        if False:
            while True:
                i = 10
        self.freezer.stop()
        self.clean_db()

    def test_run(self):
        if False:
            while True:
                i = 10
        task = LatestOnlyOperator(task_id='latest', dag=self.dag)
        task.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_skipping_non_latest(self):
        if False:
            i = 10
            return i + 15
        latest_task = LatestOnlyOperator(task_id='latest', dag=self.dag)
        downstream_task = EmptyOperator(task_id='downstream', dag=self.dag)
        downstream_task2 = EmptyOperator(task_id='downstream_2', dag=self.dag)
        downstream_task3 = EmptyOperator(task_id='downstream_3', trigger_rule=TriggerRule.NONE_FAILED, dag=self.dag)
        downstream_task.set_upstream(latest_task)
        downstream_task2.set_upstream(downstream_task)
        downstream_task3.set_upstream(downstream_task)
        self.dag.create_dagrun(run_type=DagRunType.SCHEDULED, start_date=timezone.utcnow(), execution_date=DEFAULT_DATE, state=State.RUNNING)
        self.dag.create_dagrun(run_type=DagRunType.SCHEDULED, start_date=timezone.utcnow(), execution_date=timezone.datetime(2016, 1, 1, 12), state=State.RUNNING)
        self.dag.create_dagrun(run_type=DagRunType.SCHEDULED, start_date=timezone.utcnow(), execution_date=END_DATE, state=State.RUNNING)
        latest_task.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        downstream_task.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        downstream_task2.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        downstream_task3.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        latest_instances = get_task_instances('latest')
        exec_date_to_latest_state = {ti.execution_date: ti.state for ti in latest_instances}
        assert {timezone.datetime(2016, 1, 1): 'success', timezone.datetime(2016, 1, 1, 12): 'success', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_latest_state
        downstream_instances = get_task_instances('downstream')
        exec_date_to_downstream_state = {ti.execution_date: ti.state for ti in downstream_instances}
        assert {timezone.datetime(2016, 1, 1): 'skipped', timezone.datetime(2016, 1, 1, 12): 'skipped', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_downstream_state
        downstream_instances = get_task_instances('downstream_2')
        exec_date_to_downstream_state = {ti.execution_date: ti.state for ti in downstream_instances}
        assert {timezone.datetime(2016, 1, 1): None, timezone.datetime(2016, 1, 1, 12): None, timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_downstream_state
        downstream_instances = get_task_instances('downstream_3')
        exec_date_to_downstream_state = {ti.execution_date: ti.state for ti in downstream_instances}
        assert {timezone.datetime(2016, 1, 1): 'success', timezone.datetime(2016, 1, 1, 12): 'success', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_downstream_state

    def test_not_skipping_external(self):
        if False:
            i = 10
            return i + 15
        latest_task = LatestOnlyOperator(task_id='latest', dag=self.dag)
        downstream_task = EmptyOperator(task_id='downstream', dag=self.dag)
        downstream_task2 = EmptyOperator(task_id='downstream_2', dag=self.dag)
        downstream_task.set_upstream(latest_task)
        downstream_task2.set_upstream(downstream_task)
        self.dag.create_dagrun(run_type=DagRunType.MANUAL, start_date=timezone.utcnow(), execution_date=DEFAULT_DATE, state=State.RUNNING, external_trigger=True)
        self.dag.create_dagrun(run_type=DagRunType.MANUAL, start_date=timezone.utcnow(), execution_date=timezone.datetime(2016, 1, 1, 12), state=State.RUNNING, external_trigger=True)
        self.dag.create_dagrun(run_type=DagRunType.MANUAL, start_date=timezone.utcnow(), execution_date=END_DATE, state=State.RUNNING, external_trigger=True)
        latest_task.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        downstream_task.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        downstream_task2.run(start_date=DEFAULT_DATE, end_date=END_DATE)
        latest_instances = get_task_instances('latest')
        exec_date_to_latest_state = {ti.execution_date: ti.state for ti in latest_instances}
        assert {timezone.datetime(2016, 1, 1): 'success', timezone.datetime(2016, 1, 1, 12): 'success', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_latest_state
        downstream_instances = get_task_instances('downstream')
        exec_date_to_downstream_state = {ti.execution_date: ti.state for ti in downstream_instances}
        assert {timezone.datetime(2016, 1, 1): 'success', timezone.datetime(2016, 1, 1, 12): 'success', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_downstream_state
        downstream_instances = get_task_instances('downstream_2')
        exec_date_to_downstream_state = {ti.execution_date: ti.state for ti in downstream_instances}
        assert {timezone.datetime(2016, 1, 1): 'success', timezone.datetime(2016, 1, 1, 12): 'success', timezone.datetime(2016, 1, 2): 'success'} == exec_date_to_downstream_state