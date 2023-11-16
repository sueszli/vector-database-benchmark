import typing

def f(x: typing.List[str]) -> None:
    if False:
        i = 10
        return i + 15
    ...
from typing import List

def f(x: List[str]) -> None:
    if False:
        while True:
            i = 10
    ...
import typing as t

def f(x: t.List[str]) -> None:
    if False:
        print('Hello World!')
    ...
from typing import List as IList

def f(x: IList[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: 'List[str]') -> None:
    if False:
        print('Hello World!')
    ...

def f(x: 'List[str]') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: 'List[str]') -> None:
    if False:
        print('Hello World!')
    ...

def f(x: 'List[str]') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: 'List[str]') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: "List['List[str]']") -> None:
    if False:
        i = 10
        return i + 15
    ...

def f(x: "List['Li' 'st[str]']") -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: "List['List[str]']") -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f(x: typing.Deque[str]) -> None:
    if False:
        print('Hello World!')
    ...

def f(x: typing.DefaultDict[str, str]) -> None:
    if False:
        while True:
            i = 10
    ...