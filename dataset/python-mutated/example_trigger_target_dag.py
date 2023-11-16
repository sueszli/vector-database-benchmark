"""
Example usage of the TriggerDagRunOperator. This example holds 2 DAGs:
1. 1st DAG (example_trigger_controller_dag) holds a TriggerDagRunOperator, which will trigger the 2nd DAG
2. 2nd DAG (example_trigger_target_dag) which will be triggered by the TriggerDagRunOperator in the 1st DAG
"""
from __future__ import annotations
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

@task(task_id='run_this')
def run_this_func(dag_run=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print the payload "message" passed to the DagRun conf attribute.\n\n    :param dag_run: The DagRun object\n    '
    print(f"Remotely received value of {dag_run.conf.get('message')} for key=message")
with DAG(dag_id='example_trigger_target_dag', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, schedule=None, tags=['example']) as dag:
    run_this = run_this_func()
    bash_task = BashOperator(task_id='bash_task', bash_command='echo "Here is the message: $message"', env={'message': '{{ dag_run.conf.get("message") }}'})