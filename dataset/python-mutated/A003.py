class MyClass:
    ImportError = 4
    id: int
    dir = '/'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.float = 5
        self.id = 10
        self.dir = '.'

    def str(self):
        if False:
            return 10
        pass
from typing import TypedDict

class MyClass(TypedDict):
    id: int
from threading import Event

class CustomEvent(Event):

    def set(self) -> None:
        if False:
            print('Hello World!')
        ...

    def str(self) -> None:
        if False:
            print('Hello World!')
        ...
from logging import Filter, LogRecord

class CustomFilter(Filter):

    def filter(self, record: LogRecord) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    def str(self) -> None:
        if False:
            while True:
                i = 10
        ...
from typing_extensions import override

class MyClass:

    @override
    def str(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def int(self):
        if False:
            for i in range(10):
                print('nop')
        pass