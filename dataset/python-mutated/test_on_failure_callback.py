from __future__ import annotations
import os
from datetime import datetime
from airflow.exceptions import AirflowFailException
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
DEFAULT_DATE = datetime(2016, 1, 1)
args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
dag = DAG(dag_id='test_on_failure_callback', default_args=args)

def write_data_to_callback(context):
    if False:
        return 10
    msg = ' '.join([str(k) for k in context['ti'].key.primary]) + f' fired callback with pid: {os.getpid()}'
    with open(os.environ.get('AIRFLOW_CALLBACK_FILE'), 'a+') as f:
        f.write(msg)

def task_function(ti):
    if False:
        i = 10
        return i + 15
    raise AirflowFailException()
PythonOperator(task_id='test_on_failure_callback_task', on_failure_callback=write_data_to_callback, python_callable=task_function, dag=dag)
BashOperator(task_id='bash_sleep', on_failure_callback=write_data_to_callback, bash_command='touch $AIRFLOW_CALLBACK_FILE; sleep 10', dag=dag)