from __future__ import annotations
from typing import Any
from airflow.listeners import hookimpl
from airflow.utils.state import TaskInstanceState
started_component: Any = None
stopped_component: Any = None
state: list[Any] = []

@hookimpl
def on_starting(component):
    if False:
        while True:
            i = 10
    global started_component
    started_component = component

@hookimpl
def before_stopping(component):
    if False:
        for i in range(10):
            print('nop')
    global stopped_component
    stopped_component = component

@hookimpl
def on_task_instance_running(previous_state, task_instance, session):
    if False:
        while True:
            i = 10
    state.append(TaskInstanceState.RUNNING)

@hookimpl
def on_task_instance_success(previous_state, task_instance, session):
    if False:
        return 10
    state.append(TaskInstanceState.SUCCESS)

@hookimpl
def on_task_instance_failed(previous_state, task_instance, session):
    if False:
        return 10
    state.append(TaskInstanceState.FAILED)

def clear():
    if False:
        while True:
            i = 10
    global started_component, stopped_component, state
    started_component = None
    stopped_component = None
    state = []