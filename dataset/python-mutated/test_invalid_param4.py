from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
with DAG('test_invalid_param4', start_date=datetime(2021, 1, 1), schedule='0 0 * * *', params={'str_param': Param(default='banana', type='string', minLength=2, maxLength=4)}) as the_dag:

    def print_these(*params):
        if False:
            for i in range(10):
                print('nop')
        for param in params:
            print(param)
    PythonOperator(task_id='ref_params', python_callable=print_these, op_args=['{{ params.str_param }}'])