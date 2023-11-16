from __future__ import annotations
from airflow.listeners import hookimpl
from airflow.utils.state import DagRunState, TaskInstanceState

class ClassBasedListener:

    def __init__(self):
        if False:
            return 10
        self.started_component = None
        self.stopped_component = None
        self.state = []

    @hookimpl
    def on_starting(self, component):
        if False:
            while True:
                i = 10
        self.started_component = component
        self.state.append(DagRunState.RUNNING)

    @hookimpl
    def before_stopping(self, component):
        if False:
            for i in range(10):
                print('nop')
        global stopped_component
        stopped_component = component
        self.state.append(DagRunState.SUCCESS)

    @hookimpl
    def on_task_instance_running(self, previous_state, task_instance, session):
        if False:
            print('Hello World!')
        self.state.append(TaskInstanceState.RUNNING)

    @hookimpl
    def on_task_instance_success(self, previous_state, task_instance, session):
        if False:
            print('Hello World!')
        self.state.append(TaskInstanceState.SUCCESS)

    @hookimpl
    def on_task_instance_failed(self, previous_state, task_instance, session):
        if False:
            for i in range(10):
                print('nop')
        self.state.append(TaskInstanceState.FAILED)

def clear():
    if False:
        for i in range(10):
            print('nop')
    pass