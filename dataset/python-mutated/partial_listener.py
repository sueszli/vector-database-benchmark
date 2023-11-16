from __future__ import annotations
from airflow.listeners import hookimpl
from airflow.utils.state import State
state: list[State] = []

@hookimpl
def on_task_instance_running(previous_state, task_instance, session):
    if False:
        return 10
    state.append(State.RUNNING)

def clear():
    if False:
        print('Hello World!')
    pass