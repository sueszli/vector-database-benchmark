from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.listeners import hookimpl
if TYPE_CHECKING:
    from airflow.models.dagrun import DagRun
    from airflow.models.taskinstance import TaskInstance
    from airflow.utils.state import TaskInstanceState

@hookimpl
def on_task_instance_running(previous_state: TaskInstanceState, task_instance: TaskInstance, session):
    if False:
        for i in range(10):
            print('nop')
    '\n    This method is called when task state changes to RUNNING.\n    Through callback, parameters like previous_task_state, task_instance object can be accessed.\n    This will give more information about current task_instance that is running its dag_run,\n    task and dag information.\n    '
    print('Task instance is in running state')
    print(' Previous state of the Task instance:', previous_state)
    state: TaskInstanceState = task_instance.state
    name: str = task_instance.task_id
    start_date = task_instance.start_date
    dagrun = task_instance.dag_run
    dagrun_status = dagrun.state
    task = task_instance.task
    dag = task.dag
    dag_name = None
    if dag:
        dag_name = dag.dag_id
    print(f'Current task name:{name} state:{state} start_date:{start_date}')
    print(f'Dag name:{dag_name} and current dag run status:{dagrun_status}')

@hookimpl
def on_task_instance_success(previous_state: TaskInstanceState, task_instance: TaskInstance, session):
    if False:
        for i in range(10):
            print('nop')
    '\n    This method is called when task state changes to SUCCESS.\n    Through callback, parameters like previous_task_state, task_instance object can be accessed.\n    This will give more information about current task_instance that has succeeded its\n    dag_run, task and dag information.\n    '
    print('Task instance in success state')
    print(' Previous state of the Task instance:', previous_state)
    dag_id = task_instance.dag_id
    hostname = task_instance.hostname
    operator = task_instance.operator
    dagrun = task_instance.dag_run
    queued_at = dagrun.queued_at
    print(f'Dag name:{dag_id} queued_at:{queued_at}')
    print(f'Task hostname:{hostname} operator:{operator}')

@hookimpl
def on_task_instance_failed(previous_state: TaskInstanceState, task_instance: TaskInstance, session):
    if False:
        while True:
            i = 10
    '\n    This method is called when task state changes to FAILED.\n    Through callback, parameters like previous_task_state, task_instance object can be accessed.\n    This will give more information about current task_instance that has failed its dag_run,\n    task and dag information.\n    '
    print('Task instance in failure state')
    start_date = task_instance.start_date
    end_date = task_instance.end_date
    duration = task_instance.duration
    dagrun = task_instance.dag_run
    task = task_instance.task
    dag = task_instance.task.dag
    print(f'Task start:{start_date} end:{end_date} duration:{duration}')
    print(f'Task:{task} dag:{dag} dagrun:{dagrun}')

@hookimpl
def on_dag_run_success(dag_run: DagRun, msg: str):
    if False:
        print('Hello World!')
    '\n    This method is called when dag run state changes to SUCCESS.\n    '
    print('Dag run in success state')
    start_date = dag_run.start_date
    end_date = dag_run.end_date
    print(f'Dag run start:{start_date} end:{end_date}')

@hookimpl
def on_dag_run_failed(dag_run: DagRun, msg: str):
    if False:
        while True:
            i = 10
    '\n    This method is called when dag run state changes to FAILED.\n    '
    print('Dag run  in failure state')
    dag_id = dag_run.dag_id
    run_id = dag_run.run_id
    external_trigger = dag_run.external_trigger
    print(f'Dag information:{dag_id} Run id: {run_id} external trigger: {external_trigger}')

@hookimpl
def on_dag_run_running(dag_run: DagRun, msg: str):
    if False:
        while True:
            i = 10
    '\n    This method is called when dag run state changes to RUNNING.\n    '
    print('Dag run  in running state')
    queued_at = dag_run.queued_at
    dag_hash_info = dag_run.dag_hash
    print(f'Dag information Queued at: {queued_at} hash info: {dag_hash_info}')