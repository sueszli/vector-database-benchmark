"""This is a test"""
from __future__ import nested_scopes
import site

def f(x):
    if False:
        return 10

    def g(y):
        if False:
            return 10
        return x + y
    return g
result = f(2)(4)