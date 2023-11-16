"""This is a test"""
"this isn't a doc string"
from __future__ import nested_scopes

def f(x):
    if False:
        return 10

    def g(y):
        if False:
            print('Hello World!')
        return x + y
    return g
result = f(2)(4)