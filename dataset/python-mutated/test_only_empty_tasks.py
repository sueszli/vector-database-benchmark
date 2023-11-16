from __future__ import annotations
from datetime import datetime
from typing import Sequence
from airflow.datasets import Dataset
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
DEFAULT_DATE = datetime(2016, 1, 1)
default_args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
dag = DAG(dag_id='test_only_empty_tasks', default_args=default_args, schedule='@once')

class MyEmptyOperator(EmptyOperator):
    template_fields_renderers = {'body': 'json'}
    template_fields: Sequence[str] = ('body',)

    def __init__(self, body, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.body = body
with dag:
    task_a = EmptyOperator(task_id='test_task_a')
    task_b = EmptyOperator(task_id='test_task_b')
    task_a >> task_b
    MyEmptyOperator(task_id='test_task_c', body={'hello': 'world'})
    EmptyOperator(task_id='test_task_on_execute', on_execute_callback=lambda *args, **kwargs: None)
    EmptyOperator(task_id='test_task_on_success', on_success_callback=lambda *args, **kwargs: None)
    EmptyOperator(task_id='test_task_outlets', outlets=[Dataset('hello')])