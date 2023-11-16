from __future__ import annotations
from datetime import timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.timezone import datetime
DEFAULT_DATE = datetime(2016, 1, 1)
default_args = dict(start_date=DEFAULT_DATE, owner='airflow')

def fail():
    if False:
        for i in range(10):
            print('nop')
    raise ValueError('Expected failure.')

def success(ti=None, *args, **kwargs):
    if False:
        print('Hello World!')
    if ti.execution_date != DEFAULT_DATE + timedelta(days=1):
        fail()
dag1 = DAG(dag_id='test_run_ignores_all_dependencies', default_args=dict(depends_on_past=True, **default_args))
dag1_task1 = PythonOperator(task_id='test_run_dependency_task', python_callable=fail, dag=dag1)
dag1_task2 = PythonOperator(task_id='test_run_dependent_task', python_callable=success, dag=dag1)
dag1_task1.set_downstream(dag1_task2)