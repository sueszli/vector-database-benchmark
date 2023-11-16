"""Example DAG demonstrating the usage of setup and teardown tasks."""
from __future__ import annotations
import pendulum
from airflow.decorators import setup, task, task_group, teardown
from airflow.models.dag import DAG
with DAG(dag_id='example_setup_teardown_taskflow', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:

    @task
    def my_first_task():
        if False:
            while True:
                i = 10
        print('Hello 1')

    @task
    def my_second_task():
        if False:
            return 10
        print('Hello 2')

    @task
    def my_third_task():
        if False:
            print('Hello World!')
        print('Hello 3')
    task_1 = my_first_task()
    task_2 = my_second_task()
    task_3 = my_third_task()
    task_1 >> task_2 >> task_3.as_teardown(setups=task_1)

    @setup
    def outer_setup():
        if False:
            while True:
                i = 10
        print('I am outer_setup')
        return 'some cluster id'

    @teardown
    def outer_teardown(cluster_id):
        if False:
            print('Hello World!')
        print('I am outer_teardown')
        print(f'Tearing down cluster: {cluster_id}')

    @task
    def outer_work():
        if False:
            i = 10
            return i + 15
        print('I am just a normal task')

    @task_group
    def section_1():
        if False:
            while True:
                i = 10

        @setup
        def inner_setup():
            if False:
                print('Hello World!')
            print('I set up')
            return 'some_cluster_id'

        @task
        def inner_work(cluster_id):
            if False:
                return 10
            print(f'doing some work with cluster_id={cluster_id!r}')

        @teardown
        def inner_teardown(cluster_id):
            if False:
                i = 10
                return i + 15
            print(f'tearing down cluster_id={cluster_id!r}')
        inner_setup_task = inner_setup()
        inner_work(inner_setup_task) >> inner_teardown(inner_setup_task)
    with outer_teardown(outer_setup()):
        outer_work()
        section_1()