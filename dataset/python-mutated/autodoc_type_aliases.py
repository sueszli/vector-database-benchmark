from __future__ import annotations
import io
from typing import Optional, overload
myint = int
variable: myint
variable2 = None
variable3: Optional[myint]

def read(r: io.BytesIO) -> io.StringIO:
    if False:
        for i in range(10):
            print('nop')
    'docstring'

def sum(x: myint, y: myint) -> myint:
    if False:
        while True:
            i = 10
    'docstring'
    return x + y

@overload
def mult(x: myint, y: myint) -> myint:
    if False:
        i = 10
        return i + 15
    ...

@overload
def mult(x: float, y: float) -> float:
    if False:
        print('Hello World!')
    ...

def mult(x, y):
    if False:
        for i in range(10):
            print('nop')
    'docstring'
    return (x, y)

class Foo:
    """docstring"""
    attr1: myint

    def __init__(self):
        if False:
            print('Hello World!')
        self.attr2: myint = None