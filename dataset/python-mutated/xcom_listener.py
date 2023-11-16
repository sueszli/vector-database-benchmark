from __future__ import annotations
from airflow.listeners import hookimpl

class XComListener:

    def __init__(self, path: str, task_id: str):
        if False:
            i = 10
            return i + 15
        self.path = path
        self.task_id = task_id

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
        task_instance.xcom_push(key='listener', value='listener')
        task_instance.xcom_pull(task_ids=task_instance.task_id, key='listener')
        self.write('on_task_instance_running')

    @hookimpl
    def on_task_instance_success(self, previous_state, task_instance, session):
        if False:
            for i in range(10):
                print('nop')
        read = task_instance.xcom_pull(task_ids=self.task_id, key='listener')
        self.write('on_task_instance_success')
        self.write(read)

def clear():
    if False:
        for i in range(10):
            print('nop')
    pass