from typing import List
import fastapi
import custom
from fastapi import Query

def okay(db=fastapi.Depends(get_db)):
    if False:
        print('Hello World!')
    ...

def okay(data: List[str]=fastapi.Query(None)):
    if False:
        for i in range(10):
            print('nop')
    ...

def okay(data: List[str]=Query(None)):
    if False:
        print('Hello World!')
    ...

def okay(data: custom.ImmutableTypeA=foo()):
    if False:
        i = 10
        return i + 15
    ...

def error_due_to_missing_import(data: List[str]=Depends(None)):
    if False:
        while True:
            i = 10
    ...