"""
A DAG with subdag for testing purpose.
"""
from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.subdag import SubDagOperator
DAG_NAME = 'test_subdag_operator'
DEFAULT_TASK_ARGS = {'owner': 'airflow', 'start_date': datetime(2019, 1, 1), 'max_active_runs': 1}

def subdag(parent_dag_name, child_dag_name, args):
    if False:
        i = 10
        return i + 15
    '\n    Create a subdag.\n    '
    dag_subdag = DAG(dag_id=f'{parent_dag_name}.{child_dag_name}', default_args=args, schedule='@daily')
    for i in range(2):
        EmptyOperator(task_id=f'{child_dag_name}-task-{i + 1}', default_args=args, dag=dag_subdag)
    return dag_subdag
with DAG(dag_id=DAG_NAME, start_date=datetime(2019, 1, 1), max_active_runs=1, default_args=DEFAULT_TASK_ARGS, schedule=timedelta(minutes=1)):
    start = EmptyOperator(task_id='start')
    with warnings.catch_warnings(record=True):
        section_1 = SubDagOperator(task_id='section-1', subdag=subdag(DAG_NAME, 'section-1', DEFAULT_TASK_ARGS), default_args=DEFAULT_TASK_ARGS)
    some_other_task = EmptyOperator(task_id='some-other-task')
    start >> section_1 >> some_other_task