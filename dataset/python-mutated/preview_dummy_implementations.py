from typing import NoReturn, Protocol, Union, overload

def dummy(a):
    if False:
        i = 10
        return i + 15
    ...

def other(b):
    if False:
        return 10
    ...

@overload
def a(arg: int) -> int:
    if False:
        i = 10
        return i + 15
    ...

@overload
def a(arg: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def a(arg: object) -> NoReturn:
    if False:
        i = 10
        return i + 15
    ...

def a(arg: Union[int, str, object]) -> Union[int, str]:
    if False:
        print('Hello World!')
    if not isinstance(arg, (int, str)):
        raise TypeError
    return arg

class Proto(Protocol):

    def foo(self, a: int) -> int:
        if False:
            return 10
        ...

    def bar(self, b: str) -> str:
        if False:
            return 10
        ...

    def baz(self, c: bytes) -> str:
        if False:
            while True:
                i = 10
        ...

def dummy_two():
    if False:
        i = 10
        return i + 15
    ...

@dummy
def dummy_three():
    if False:
        i = 10
        return i + 15
    ...

def dummy_four():
    if False:
        while True:
            i = 10
    ...

@overload
def b(arg: int) -> int:
    if False:
        while True:
            i = 10
    ...

@overload
def b(arg: str) -> str:
    if False:
        return 10
    ...

@overload
def b(arg: object) -> NoReturn:
    if False:
        while True:
            i = 10
    ...

def b(arg: Union[int, str, object]) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    if not isinstance(arg, (int, str)):
        raise TypeError
    return arg
from typing import NoReturn, Protocol, Union, overload

def dummy(a):
    if False:
        i = 10
        return i + 15
    ...

def other(b):
    if False:
        i = 10
        return i + 15
    ...

@overload
def a(arg: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def a(arg: str) -> str:
    if False:
        print('Hello World!')
    ...

@overload
def a(arg: object) -> NoReturn:
    if False:
        return 10
    ...

def a(arg: Union[int, str, object]) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(arg, (int, str)):
        raise TypeError
    return arg

class Proto(Protocol):

    def foo(self, a: int) -> int:
        if False:
            return 10
        ...

    def bar(self, b: str) -> str:
        if False:
            print('Hello World!')
        ...

    def baz(self, c: bytes) -> str:
        if False:
            print('Hello World!')
        ...

def dummy_two():
    if False:
        i = 10
        return i + 15
    ...

@dummy
def dummy_three():
    if False:
        return 10
    ...

def dummy_four():
    if False:
        while True:
            i = 10
    ...

@overload
def b(arg: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def b(arg: str) -> str:
    if False:
        while True:
            i = 10
    ...

@overload
def b(arg: object) -> NoReturn:
    if False:
        print('Hello World!')
    ...

def b(arg: Union[int, str, object]) -> Union[int, str]:
    if False:
        while True:
            i = 10
    if not isinstance(arg, (int, str)):
        raise TypeError
    return arg