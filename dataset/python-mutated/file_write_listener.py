from __future__ import annotations
import logging
from airflow.cli.commands.task_command import TaskCommandMarker
from airflow.listeners import hookimpl
log = logging.getLogger(__name__)

class FileWriteListener:

    def __init__(self, path):
        if False:
            while True:
                i = 10
        self.path = path

    def write(self, line: str):
        if False:
            print('Hello World!')
        with open(self.path, 'a') as f:
            f.write(line + '\n')

    @hookimpl
    def on_task_instance_running(self, previous_state, task_instance, session):
        if False:
            for i in range(10):
                print('nop')
        self.write('on_task_instance_running')

    @hookimpl
    def on_task_instance_success(self, previous_state, task_instance, session):
        if False:
            return 10
        self.write('on_task_instance_success')

    @hookimpl
    def on_task_instance_failed(self, previous_state, task_instance, session):
        if False:
            while True:
                i = 10
        self.write('on_task_instance_failed')

    @hookimpl
    def on_starting(self, component):
        if False:
            i = 10
            return i + 15
        if isinstance(component, TaskCommandMarker):
            self.write('on_starting')

    @hookimpl
    def before_stopping(self, component):
        if False:
            return 10
        if isinstance(component, TaskCommandMarker):
            self.write('before_stopping')

def clear():
    if False:
        while True:
            i = 10
    pass