from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
import time_machine
from airflow import settings
from airflow.models import DagRun, TaskInstance
from airflow.ti_deps.deps.runnable_exec_date_dep import RunnableExecDateDep
from airflow.utils.timezone import datetime
from airflow.utils.types import DagRunType
pytestmark = pytest.mark.db_test

@pytest.fixture(autouse=True, scope='function')
def clean_db(session):
    if False:
        for i in range(10):
            print('nop')
    yield
    session.query(DagRun).delete()
    session.query(TaskInstance).delete()

@time_machine.travel('2016-11-01')
@pytest.mark.parametrize('allow_trigger_in_future,schedule,execution_date,is_met', [(True, None, datetime(2016, 11, 3), True), (True, '@daily', datetime(2016, 11, 3), False), (False, None, datetime(2016, 11, 3), False), (False, '@daily', datetime(2016, 11, 3), False), (False, '@daily', datetime(2016, 11, 1), True), (False, None, datetime(2016, 11, 1), True)])
def test_exec_date_dep(dag_maker, session, create_dummy_dag, allow_trigger_in_future, schedule, execution_date, is_met):
    if False:
        while True:
            i = 10
    "\n    If the dag's execution date is in the future but (allow_trigger_in_future=False or not schedule)\n    this dep should fail\n    "
    with patch.object(settings, 'ALLOW_FUTURE_EXEC_DATES', allow_trigger_in_future):
        create_dummy_dag('test_localtaskjob_heartbeat', start_date=datetime(2015, 1, 1), end_date=datetime(2016, 11, 5), schedule=schedule, with_dagrun_type=DagRunType.MANUAL, session=session)
        (ti,) = dag_maker.create_dagrun(execution_date=execution_date).task_instances
        assert RunnableExecDateDep().is_met(ti=ti) == is_met

@time_machine.travel('2016-01-01')
def test_exec_date_after_end_date(session, dag_maker, create_dummy_dag):
    if False:
        return 10
    "\n    If the dag's execution date is in the future this dep should fail\n    "
    create_dummy_dag('test_localtaskjob_heartbeat', start_date=datetime(2015, 1, 1), end_date=datetime(2016, 11, 5), schedule=None, with_dagrun_type=DagRunType.MANUAL, session=session)
    (ti,) = dag_maker.create_dagrun(execution_date=datetime(2016, 11, 2)).task_instances
    assert not RunnableExecDateDep().is_met(ti=ti)

class TestRunnableExecDateDep:

    def _get_task_instance(self, execution_date, dag_end_date=None, task_end_date=None):
        if False:
            i = 10
            return i + 15
        dag = Mock(end_date=dag_end_date)
        dagrun = Mock(execution_date=execution_date)
        task = Mock(dag=dag, end_date=task_end_date)
        return Mock(task=task, get_dagrun=Mock(return_value=dagrun))

    def test_exec_date_after_task_end_date(self):
        if False:
            while True:
                i = 10
        '\n        If the task instance execution date is after the tasks end date\n        this dep should fail\n        '
        ti = self._get_task_instance(dag_end_date=datetime(2016, 1, 3), task_end_date=datetime(2016, 1, 1), execution_date=datetime(2016, 1, 2))
        assert not RunnableExecDateDep().is_met(ti=ti)

    def test_exec_date_after_dag_end_date(self):
        if False:
            return 10
        "\n        If the task instance execution date is after the dag's end date\n        this dep should fail\n        "
        ti = self._get_task_instance(dag_end_date=datetime(2016, 1, 1), task_end_date=datetime(2016, 1, 3), execution_date=datetime(2016, 1, 2))
        assert not RunnableExecDateDep().is_met(ti=ti)

    def test_all_deps_met(self):
        if False:
            return 10
        '\n        Test to make sure all the conditions for the dep are met\n        '
        ti = self._get_task_instance(dag_end_date=datetime(2016, 1, 2), task_end_date=datetime(2016, 1, 2), execution_date=datetime(2016, 1, 1))
        assert RunnableExecDateDep().is_met(ti=ti)