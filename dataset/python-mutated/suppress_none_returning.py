"""Test case expected to be run with `suppress_none_returning = True`."""

def foo():
    if False:
        return 10
    a = 2 + 2

def foo():
    if False:
        while True:
            i = 10
    return

def foo():
    if False:
        while True:
            i = 10
    return None

def foo():
    if False:
        i = 10
        return i + 15
    a = 2 + 2
    if a == 4:
        return
    else:
        return

def foo():
    if False:
        for i in range(10):
            print('nop')
    a = 2 + 2
    if a == 4:
        return None
    else:
        return

def foo():
    if False:
        return 10

    def bar() -> bool:
        if False:
            i = 10
            return i + 15
        return True
    bar()

def foo():
    if False:
        while True:
            i = 10
    return True

def foo():
    if False:
        print('Hello World!')
    a = 2 + 2
    if a == 4:
        return True
    else:
        return

def foo(a):
    if False:
        return 10
    a = 2 + 2