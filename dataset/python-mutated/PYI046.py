import typing
from typing import Protocol

class _Foo(Protocol):
    bar: int

class _Bar(typing.Protocol):
    bar: int

class _UsedPrivateProtocol(Protocol):
    bar: int

def uses__UsedPrivateProtocol(arg: _UsedPrivateProtocol) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...