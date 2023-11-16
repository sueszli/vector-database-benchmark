from __future__ import annotations
import pendulum
import pytest
import time_machine
from airflow.models import Log, TaskInstance
from airflow.operators.empty import EmptyOperator
from airflow.utils import timezone
from airflow.utils.session import provide_session
from airflow.utils.state import State
from tests.test_utils.db import clear_db_dags, clear_db_logs, clear_db_runs
pytestmark = pytest.mark.db_test

@pytest.fixture(autouse=True)
def clear_db():
    if False:
        for i in range(10):
            print('nop')
    clear_db_logs()
    clear_db_runs()
    clear_db_dags()
    yield

def add_log(execdate, session, dag_maker, timezone_override=None):
    if False:
        i = 10
        return i + 15
    with dag_maker(dag_id='logging', default_args={'start_date': execdate}):
        task = EmptyOperator(task_id='dummy')
    dag_maker.create_dagrun()
    task_instance = TaskInstance(task=task, execution_date=execdate, state='success')
    session.merge(task_instance)
    log = Log(State.RUNNING, task_instance)
    if timezone_override:
        log.dttm = log.dttm.astimezone(timezone_override)
    session.add(log)
    session.commit()
    return log

@provide_session
def test_timestamp_behaviour(dag_maker, session=None):
    if False:
        i = 10
        return i + 15
    execdate = timezone.utcnow()
    with time_machine.travel(execdate, tick=False):
        current_time = timezone.utcnow()
        old_log = add_log(execdate, session, dag_maker)
        session.expunge(old_log)
        log_time = session.query(Log).one().dttm
        assert log_time == current_time
        assert log_time.tzinfo.name == 'UTC'

@provide_session
def test_timestamp_behaviour_with_timezone(dag_maker, session=None):
    if False:
        while True:
            i = 10
    execdate = timezone.utcnow()
    with time_machine.travel(execdate, tick=False):
        current_time = timezone.utcnow()
        old_log = add_log(execdate, session, dag_maker, timezone_override=pendulum.timezone('Europe/Warsaw'))
        session.expunge(old_log)
        log_time = session.query(Log).one().dttm
        assert log_time == current_time
        assert old_log.dttm.tzinfo.name != 'UTC'
        assert log_time.tzinfo.name == 'UTC'
        assert old_log.dttm.astimezone(pendulum.timezone('UTC')) == log_time