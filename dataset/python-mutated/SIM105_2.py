"""Case: `contextlib` already imported."""
import contextlib

def foo():
    if False:
        for i in range(10):
            print('nop')
    pass
try:
    foo()
except ValueError:
    pass