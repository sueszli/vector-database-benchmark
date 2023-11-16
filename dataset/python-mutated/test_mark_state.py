from __future__ import annotations
from datetime import datetime
from time import sleep
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.session import create_session
from airflow.utils.state import State
DEFAULT_DATE = datetime(2016, 1, 1)
args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
dag_id = 'test_mark_state'
dag = DAG(dag_id=dag_id, default_args=args)

def success_callback(context):
    if False:
        i = 10
        return i + 15
    assert context['dag_run'].dag_id == dag_id

def test_mark_success_no_kill(ti):
    if False:
        print('Hello World!')
    assert ti.state == State.RUNNING
    with create_session() as session:
        ti.state = State.SUCCESS
        session.merge(ti)
        session.commit()
        sleep(10)
PythonOperator(task_id='test_mark_success_no_kill', python_callable=test_mark_success_no_kill, dag=dag, on_success_callback=success_callback)

def check_failure(context):
    if False:
        i = 10
        return i + 15
    assert context['dag_run'].dag_id == dag_id
    assert context['exception'] == 'task marked as failed externally'

def test_mark_failure_externally(ti):
    if False:
        return 10
    assert State.RUNNING == ti.state
    with create_session() as session:
        ti.log.info("Marking TI as failed 'externally'")
        ti.state = State.FAILED
        session.merge(ti)
        session.commit()
    sleep(10)
    assert False
PythonOperator(task_id='test_mark_failure_externally', python_callable=test_mark_failure_externally, on_failure_callback=check_failure, dag=dag)

def test_mark_skipped_externally(ti):
    if False:
        return 10
    assert State.RUNNING == ti.state
    sleep(0.1)
    with create_session() as session:
        ti.log.info("Marking TI as failed 'externally'")
        ti.state = State.SKIPPED
        session.merge(ti)
        session.commit()
    sleep(10)
    assert False
PythonOperator(task_id='test_mark_skipped_externally', python_callable=test_mark_skipped_externally, dag=dag)
PythonOperator(task_id='dummy', python_callable=lambda : True, dag=dag)