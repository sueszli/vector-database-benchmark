"""Tests to determine standard library import classification.

For typing-only import detection tests, see `TCH002.py`.
"""

def f():
    if False:
        for i in range(10):
            print('nop')
    import os
    x: os

def f():
    if False:
        for i in range(10):
            print('nop')
    import os
    print(os)