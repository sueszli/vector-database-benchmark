from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
with DAG('test_valid_param', start_date=datetime(2021, 1, 1), schedule=None, params={'str_param': Param(type='string', minLength=2, maxLength=4), 'str_param2': Param(None, type='string', minLength=2, maxLength=4), 'str_param3': Param('valid_default', type='string', minLength=2, maxLength=15)}) as the_dag:

    def print_these(*params):
        if False:
            print('Hello World!')
        for param in params:
            print(param)
    PythonOperator(task_id='ref_params', python_callable=print_these, op_args=['{{ params.str_param }}'])