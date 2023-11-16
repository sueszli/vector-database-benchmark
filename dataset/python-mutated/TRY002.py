"""
Violation:

Raising vanilla exception with custom message means it should be customized.
"""
from somewhere import exceptions

def func():
    if False:
        i = 10
        return i + 15
    a = 1
    if a == 1:
        raise Exception('Custom message')
    b = 1
    if b == 1:
        raise Exception

def ignore():
    if False:
        while True:
            i = 10
    try:
        a = 1
    except Exception as ex:
        raise ex

def anotherfunc():
    if False:
        while True:
            i = 10
    a = 1
    if a == 1:
        raise exceptions.Exception('Another except')