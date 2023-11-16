"""Test that straight `__future__` imports are considered unused."""

def f():
    if False:
        for i in range(10):
            print('nop')
    import __future__

def f():
    if False:
        return 10
    import __future__
    print(__future__.absolute_import)