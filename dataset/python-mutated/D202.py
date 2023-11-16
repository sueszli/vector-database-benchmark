"""Test: no-blank-line-after-function special-cases around comment handling."""

def outer():
    if False:
        print('Hello World!')
    'This is a docstring.'

    def inner():
        if False:
            print('Hello World!')
        return
    return inner

def outer():
    if False:
        for i in range(10):
            print('nop')
    'This is a docstring.'

    def inner():
        if False:
            while True:
                i = 10
        return
    return inner

def outer():
    if False:
        i = 10
        return i + 15
    'This is a docstring.'

    def inner():
        if False:
            i = 10
            return i + 15
        return
    return inner

def outer():
    if False:
        i = 10
        return i + 15
    'This is a docstring.'

    def inner():
        if False:
            print('Hello World!')
        return
    return inner

def outer():
    if False:
        while True:
            i = 10
    'This is a docstring.'

    def inner():
        if False:
            while True:
                i = 10
        return
    return inner

def outer():
    if False:
        i = 10
        return i + 15
    'This is a docstring.'

    def inner():
        if False:
            i = 10
            return i + 15
        return
    return inner

def outer():
    if False:
        for i in range(10):
            print('nop')
    'This is a docstring.'

    def inner():
        if False:
            print('Hello World!')
        return
    return inner

def outer():
    if False:
        print('Hello World!')
    'This is a docstring.'

    def inner():
        if False:
            i = 10
            return i + 15
        return
    return inner