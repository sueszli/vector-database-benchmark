"""
DAG designed to test what happens when a DAG with pooled tasks is run
by a BackfillJob.
Addresses issue #1225.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
DEFAULT_DATE = datetime(2016, 1, 1)
default_args = dict(start_date=DEFAULT_DATE, owner='airflow')

def fail():
    if False:
        print('Hello World!')
    raise ValueError('Expected failure.')
dag1 = DAG(dag_id='test_backfill_pooled_task_dag', default_args=default_args)
dag1_task1 = EmptyOperator(task_id='test_backfill_pooled_task', dag=dag1, pool='test_backfill_pooled_task_pool')
dag3 = DAG(dag_id='test_dagrun_states_fail', default_args=default_args)
dag3_task1 = PythonOperator(task_id='test_dagrun_fail', dag=dag3, python_callable=fail)
dag3_task2 = EmptyOperator(task_id='test_dagrun_succeed', dag=dag3)
dag3_task2.set_upstream(dag3_task1)
dag4 = DAG(dag_id='test_dagrun_states_success', default_args=default_args)
dag4_task1 = PythonOperator(task_id='test_dagrun_fail', dag=dag4, python_callable=fail)
dag4_task2 = EmptyOperator(task_id='test_dagrun_succeed', dag=dag4, trigger_rule=TriggerRule.ALL_FAILED)
dag4_task2.set_upstream(dag4_task1)
dag5 = DAG(dag_id='test_dagrun_states_root_fail', default_args=default_args)
dag5_task1 = EmptyOperator(task_id='test_dagrun_succeed', dag=dag5)
dag5_task2 = PythonOperator(task_id='test_dagrun_fail', dag=dag5, python_callable=fail)
dag6 = DAG(dag_id='test_dagrun_states_deadlock', default_args=default_args)
dag6_task1 = EmptyOperator(task_id='test_depends_on_past', depends_on_past=True, dag=dag6)
dag6_task2 = EmptyOperator(task_id='test_depends_on_past_2', depends_on_past=True, dag=dag6)
dag6_task2.set_upstream(dag6_task1)
dag8 = DAG(dag_id='test_dagrun_states_root_fail_unfinished', default_args=default_args)
dag8_task1 = EmptyOperator(task_id='test_dagrun_unfinished', dag=dag8)
dag8_task2 = PythonOperator(task_id='test_dagrun_fail', dag=dag8, python_callable=fail)
dag9 = DAG(dag_id='test_dagrun_states_root_future', default_args=default_args)
dag9_task1 = EmptyOperator(task_id='current', dag=dag9)
dag9_task2 = EmptyOperator(task_id='future', dag=dag9, start_date=DEFAULT_DATE + timedelta(days=1))