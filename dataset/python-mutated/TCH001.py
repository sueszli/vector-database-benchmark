"""Tests to determine first-party import classification.

For typing-only import detection tests, see `TCH002.py`.
"""

def f():
    if False:
        i = 10
        return i + 15
    import TYP001
    x: TYP001

def f():
    if False:
        i = 10
        return i + 15
    import TYP001
    print(TYP001)

def f():
    if False:
        while True:
            i = 10
    from . import TYP001
    x: TYP001

def f():
    if False:
        while True:
            i = 10
    from . import TYP001
    print(TYP001)