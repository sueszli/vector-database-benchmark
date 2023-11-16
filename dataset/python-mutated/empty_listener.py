from __future__ import annotations
from airflow.listeners import hookimpl

@hookimpl
def on_task_instance_running(previous_state, task_instance, session):
    if False:
        print('Hello World!')
    pass

def clear():
    if False:
        return 10
    pass