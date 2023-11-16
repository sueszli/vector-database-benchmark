from typing import NoReturn, Never
import typing_extensions

def foo(arg):
    if False:
        for i in range(10):
            print('nop')
    ...

def foo_int(arg: int):
    if False:
        print('Hello World!')
    ...

def foo_no_return(arg: NoReturn):
    if False:
        return 10
    ...

def foo_no_return_typing_extensions(arg: typing_extensions.NoReturn):
    if False:
        for i in range(10):
            print('nop')
    ...

def foo_no_return_kwarg(arg: int, *, arg2: NoReturn):
    if False:
        return 10
    ...

def foo_no_return_pos_only(arg: int, /, arg2: NoReturn):
    if False:
        i = 10
        return i + 15
    ...

def foo_never(arg: Never):
    if False:
        for i in range(10):
            print('nop')
    ...