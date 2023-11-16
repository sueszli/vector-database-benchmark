"""Case: `contextlib` is imported after the call site."""

def foo():
    if False:
        print('Hello World!')
    pass

def bar():
    if False:
        for i in range(10):
            print('nop')
    try:
        foo()
    except ValueError:
        pass
import contextlib