from typing import Union

@bird
def zoo():
    if False:
        return 10
    ...

class A:
    ...

@bar
class B:

    def BMethod(self) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def BMethod(self, arg: List[str]) -> None:
        if False:
            while True:
                i = 10
        ...

class C:
    ...

@hmm
class D:
    ...

class E:
    ...

@baz
def foo() -> None:
    if False:
        i = 10
        return i + 15
    ...

class F(A, C):
    ...

def spam() -> None:
    if False:
        i = 10
        return i + 15
    ...

@overload
def spam(arg: str) -> str:
    if False:
        while True:
            i = 10
    ...
var: int = 1

def eggs() -> Union[str, int]:
    if False:
        return 10
    ...
from typing import Union

@bird
def zoo():
    if False:
        return 10
    ...

class A:
    ...

@bar
class B:

    def BMethod(self) -> None:
        if False:
            return 10
        ...

    @overload
    def BMethod(self, arg: List[str]) -> None:
        if False:
            print('Hello World!')
        ...

class C:
    ...

@hmm
class D:
    ...

class E:
    ...

@baz
def foo() -> None:
    if False:
        while True:
            i = 10
    ...

class F(A, C):
    ...

def spam() -> None:
    if False:
        print('Hello World!')
    ...

@overload
def spam(arg: str) -> str:
    if False:
        print('Hello World!')
    ...
var: int = 1

def eggs() -> Union[str, int]:
    if False:
        return 10
    ...