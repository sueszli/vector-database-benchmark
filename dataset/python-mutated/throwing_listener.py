from __future__ import annotations
from airflow.listeners import hookimpl

@hookimpl
def on_task_instance_success(previous_state, task_instance, session):
    if False:
        i = 10
        return i + 15
    raise RuntimeError()

def clear():
    if False:
        for i in range(10):
            print('nop')
    pass