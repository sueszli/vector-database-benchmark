""" These tests contain all forms of closure absuse.

"""
from __future__ import print_function
a = 1
b = 1

def someFunction():
    if False:
        while True:
            i = 10
    a = a

class SomeClass:
    b = b
SomeClass()
try:
    someFunction()
except UnboundLocalError as e:
    print('Expected unbound local error occurred:', repr(e))
try:

    class AnotherClass:
        b = undefined_global
except NameError as e:
    print('Expected name error occurred:', repr(e))
try:

    class YetAnotherClass:
        b = 1
        del b
        print(b)
except NameError as e:
    print('Expected name error occurred:', repr(e))