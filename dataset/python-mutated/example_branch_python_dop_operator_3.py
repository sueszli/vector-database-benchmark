"""
Example DAG demonstrating the usage of ``@task.branch`` TaskFlow API decorator with depends_on_past=True,
where tasks may be run or skipped on alternating runs.
"""
from __future__ import annotations
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator

@task.branch()
def should_run(**kwargs) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine which empty_task should be run based on if the execution date minute is even or odd.\n\n    :param dict kwargs: Context\n    :return: Id of the task to run\n    '
    print(f"------------- exec dttm = {kwargs['execution_date']} and minute = {kwargs['execution_date'].minute}")
    if kwargs['execution_date'].minute % 2 == 0:
        return 'empty_task_1'
    else:
        return 'empty_task_2'
with DAG(dag_id='example_branch_dop_operator_v3', schedule='*/1 * * * *', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, default_args={'depends_on_past': True}, tags=['example']) as dag:
    cond = should_run()
    empty_task_1 = EmptyOperator(task_id='empty_task_1')
    empty_task_2 = EmptyOperator(task_id='empty_task_2')
    cond >> [empty_task_1, empty_task_2]