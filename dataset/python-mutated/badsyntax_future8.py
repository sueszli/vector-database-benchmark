"""This is a test"""
from __future__ import *

def f(x):
    if False:
        print('Hello World!')

    def g(y):
        if False:
            print('Hello World!')
        return x + y
    return g
print(f(2)(4))