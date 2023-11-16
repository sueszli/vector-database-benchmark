"""This is a test"""
import __future__
from __future__ import nested_scopes

def f(x):
    if False:
        i = 10
        return i + 15

    def g(y):
        if False:
            while True:
                i = 10
        return x + y
    return g
result = f(2)(4)