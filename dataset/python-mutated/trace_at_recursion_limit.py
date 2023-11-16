"""
From http://bugs.python.org/issue6717

A misbehaving trace hook can trigger a segfault by exceeding the recursion
limit.
"""
import sys

def x():
    if False:
        for i in range(10):
            print('nop')
    pass

def g(*args):
    if False:
        i = 10
        return i + 15
    if True:
        try:
            x()
        except:
            pass
    return g

def f():
    if False:
        while True:
            i = 10
    print(sys.getrecursionlimit())
    f()
sys.settrace(g)
f()