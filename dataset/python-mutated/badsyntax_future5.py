"""This is a test"""
from __future__ import nested_scopes
import foo
from __future__ import nested_scopes

def f(x):
    if False:
        i = 10
        return i + 15

    def g(y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    return g
result = f(2)(4)