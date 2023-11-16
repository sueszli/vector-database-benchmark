from typing import Protocol

class P:

    def __call__(self, arg: str) -> str:
        if False:
            while True:
                i = 10
        ...

def returns_p() -> P:
    if False:
        while True:
            i = 10
    ...
p: P = returns_p()

def foo() -> str:
    if False:
        while True:
            i = 10
    return p('a')

class CallableProtocol(Protocol):

    def __call__(self, arg: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

def returns_callable_protocol() -> CallableProtocol:
    if False:
        i = 10
        return i + 15
    ...

def bar() -> str:
    if False:
        i = 10
        return i + 15
    p = returns_callable_protocol()
    return p('b')