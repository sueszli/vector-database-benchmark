from __future__ import annotations
from typing import Any
from airflow.listeners import hookimpl
started_component: Any = None
stopped_component: Any = None

@hookimpl
def on_starting(component):
    if False:
        for i in range(10):
            print('nop')
    global started_component
    started_component = component

@hookimpl
def before_stopping(component):
    if False:
        for i in range(10):
            print('nop')
    global stopped_component
    stopped_component = component

def clear():
    if False:
        print('Hello World!')
    global started_component, stopped_component
    started_component = None
    stopped_component = None