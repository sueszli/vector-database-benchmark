from typing import overload

@overload
def foo(i: int) -> 'int':
    if False:
        print('Hello World!')
    ...

@overload
def foo(i: 'str') -> 'str':
    if False:
        while True:
            i = 10
    ...

def foo(i):
    if False:
        while True:
            i = 10
    return i

@overload
def bar(i: int) -> 'int':
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def bar(i: 'str') -> 'str':
    if False:
        print('Hello World!')
    ...

class X:

    def bar(i):
        if False:
            i = 10
            return i + 15
        return i

@overload
def baz(i: int) -> 'int':
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def baz(i: 'str') -> 'str':
    if False:
        return 10
    ...
x = 1

def baz(i):
    if False:
        while True:
            i = 10
    return i