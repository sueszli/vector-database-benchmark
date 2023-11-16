"""
DAG designed to test a PythonOperator that calls a functool.partial
"""
from __future__ import annotations
import functools
import logging
from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
DEFAULT_DATE = datetime(2016, 1, 1)
default_args = dict(start_date=DEFAULT_DATE, owner='airflow')

class CallableClass:
    """
    Class that is callable.
    """

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        'A __call__ method'

def a_function(_, __):
    if False:
        print('Hello World!')
    'A function with two args'
partial_function = functools.partial(a_function, arg_x=1)
class_instance = CallableClass()
logging.info('class_instance type: %s', type(class_instance))
dag = DAG(dag_id='test_task_view_type_check', default_args=default_args)
dag_task1 = PythonOperator(task_id='test_dagrun_functool_partial', dag=dag, python_callable=partial_function)
dag_task2 = PythonOperator(task_id='test_dagrun_instance', dag=dag, python_callable=class_instance)