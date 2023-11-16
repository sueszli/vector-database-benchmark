from __future__ import annotations
import copy
import typing
from airflow.listeners import hookimpl
if typing.TYPE_CHECKING:
    from airflow.datasets import Dataset
changed: list[Dataset] = []
created: list[Dataset] = []

@hookimpl
def on_dataset_changed(dataset):
    if False:
        i = 10
        return i + 15
    changed.append(copy.deepcopy(dataset))

@hookimpl
def on_dataset_created(dataset):
    if False:
        while True:
            i = 10
    created.append(copy.deepcopy(dataset))

def clear():
    if False:
        for i in range(10):
            print('nop')
    global changed, created
    (changed, created) = ([], [])