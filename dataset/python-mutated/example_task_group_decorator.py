"""Example DAG demonstrating the usage of the @taskgroup decorator."""
from __future__ import annotations
import pendulum
from airflow.decorators import task, task_group
from airflow.models.dag import DAG

@task
def task_start():
    if False:
        return 10
    'Empty Task which is First Task of Dag'
    return '[Task_start]'

@task
def task_1(value: int) -> str:
    if False:
        return 10
    'Empty Task1'
    return f'[ Task1 {value} ]'

@task
def task_2(value: str) -> str:
    if False:
        i = 10
        return i + 15
    'Empty Task2'
    return f'[ Task2 {value} ]'

@task
def task_3(value: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Empty Task3'
    print(f'[ Task3 {value} ]')

@task
def task_end() -> None:
    if False:
        while True:
            i = 10
    'Empty Task which is Last Task of Dag'
    print('[ Task_End  ]')

@task_group
def task_group_function(value: int) -> None:
    if False:
        i = 10
        return i + 15
    'TaskGroup for grouping related Tasks'
    task_3(task_2(task_1(value)))
with DAG(dag_id='example_task_group_decorator', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:
    start_task = task_start()
    end_task = task_end()
    for i in range(5):
        current_task_group = task_group_function(i)
        start_task >> current_task_group >> end_task