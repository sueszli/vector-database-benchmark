"""This is a test"""
from __future__ import nested_scopes
from __future__ import rested_snopes

def f(x):
    if False:
        return 10

    def g(y):
        if False:
            i = 10
            return i + 15
        return x + y
    return g
result = f(2)(4)