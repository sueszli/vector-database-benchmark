from __future__ import annotations
from datetime import datetime
from fake_datetime import FakeDatetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
DEFAULT_DATE = datetime(2016, 1, 1)
args = {'owner': 'airflow', 'start_date': DEFAULT_DATE, 'run_as_user': 'airflow_test_user'}
dag = DAG(dag_id='impersonation_with_custom_pkg', default_args=args)

def print_today():
    if False:
        return 10
    date_time = FakeDatetime.utcnow()
    print(f'Today is {date_time:%Y-%m-%d}')

def check_hive_conf():
    if False:
        i = 10
        return i + 15
    from airflow.configuration import conf
    assert conf.get('hive', 'default_hive_mapred_queue') == 'airflow'
PythonOperator(python_callable=print_today, task_id='exec_python_fn', dag=dag)
PythonOperator(python_callable=check_hive_conf, task_id='exec_check_hive_conf_fn', dag=dag)