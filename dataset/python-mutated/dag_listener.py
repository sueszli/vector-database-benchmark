from __future__ import annotations
import copy
import typing
from airflow.listeners import hookimpl
if typing.TYPE_CHECKING:
    from airflow.models.dagrun import DagRun
running: list[DagRun] = []
success: list[DagRun] = []
failure: list[DagRun] = []

@hookimpl
def on_dag_run_running(dag_run: DagRun, msg: str):
    if False:
        print('Hello World!')
    running.append(copy.deepcopy(dag_run))

@hookimpl
def on_dag_run_success(dag_run: DagRun, msg: str):
    if False:
        while True:
            i = 10
    success.append(copy.deepcopy(dag_run))

@hookimpl
def on_dag_run_failed(dag_run: DagRun, msg: str):
    if False:
        for i in range(10):
            print('nop')
    failure.append(dag_run)

def clear():
    if False:
        for i in range(10):
            print('nop')
    global running, success, failure
    (running, success, failure) = ([], [], [])