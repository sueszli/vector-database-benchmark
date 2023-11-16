from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
with DAG('test_valid_param2', start_date=datetime(2021, 1, 1), schedule='0 0 * * *', params={'str_param': Param('some_default', type='string', minLength=2, maxLength=12), 'optional_str_param': Param(None, type=['null', 'string'])}) as the_dag:

    def print_these(*params):
        if False:
            print('Hello World!')
        for param in params:
            print(param)
    PythonOperator(task_id='ref_params', python_callable=print_these, op_args=['{{ params.str_param }}'])