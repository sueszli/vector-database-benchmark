"""Test fix for issue #8441.

Ref: https://github.com/astral-sh/ruff/issues/8441
"""

def foo():
    if False:
        i = 10
        return i + 15
    ...

def bar():
    if False:
        print('Hello World!')
    a = foo()
    (b, c) = foo()

def baz():
    if False:
        return 10
    (d, _e) = foo()
    print(d)

def qux():
    if False:
        print('Hello World!')
    (f, _) = foo()
    print(f)

def quux():
    if False:
        while True:
            i = 10
    (g, h) = foo()
    print(g, h)

def quuz():
    if False:
        for i in range(10):
            print('nop')
    (_i, _j) = foo()