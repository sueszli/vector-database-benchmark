from __future__ import annotations
import datetime
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

@task
def make_arg_lists():
    if False:
        while True:
            i = 10
    return [[1], [2], [{'a': 'b'}]]

def consumer(value):
    if False:
        return 10
    print(repr(value))
with DAG(dag_id='test_mapped_classic', start_date=datetime.datetime(2022, 1, 1)) as dag:
    PythonOperator.partial(task_id='consumer', python_callable=consumer).expand(op_args=make_arg_lists())
    PythonOperator.partial(task_id='consumer_literal', python_callable=consumer).expand(op_args=[[1], [2], [3]])