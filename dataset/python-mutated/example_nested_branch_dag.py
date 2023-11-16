"""
Example DAG demonstrating a workflow with nested branching. The join tasks are created with
``none_failed_min_one_success`` trigger rule such that they are skipped whenever their corresponding
branching tasks are skipped.
"""
from __future__ import annotations
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
with DAG(dag_id='example_nested_branch_dag', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, schedule='@daily', tags=['example']) as dag:

    @task.branch()
    def branch(task_id_to_return: str) -> str:
        if False:
            print('Hello World!')
        return task_id_to_return
    branch_1 = branch.override(task_id='branch_1')(task_id_to_return='true_1')
    join_1 = EmptyOperator(task_id='join_1', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    true_1 = EmptyOperator(task_id='true_1')
    false_1 = EmptyOperator(task_id='false_1')
    branch_2 = branch.override(task_id='branch_2')(task_id_to_return='true_2')
    join_2 = EmptyOperator(task_id='join_2', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    true_2 = EmptyOperator(task_id='true_2')
    false_2 = EmptyOperator(task_id='false_2')
    false_3 = EmptyOperator(task_id='false_3')
    branch_1 >> true_1 >> join_1
    branch_1 >> false_1 >> branch_2 >> [true_2, false_2] >> join_2 >> false_3 >> join_1