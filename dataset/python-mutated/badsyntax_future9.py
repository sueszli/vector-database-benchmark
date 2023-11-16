"""This is a test"""
from __future__ import nested_scopes, braces

def f(x):
    if False:
        while True:
            i = 10

    def g(y):
        if False:
            return 10
        return x + y
    return g
print(f(2)(4))