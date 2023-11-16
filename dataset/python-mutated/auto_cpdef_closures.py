def closure_func(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> c = closure_func(2)\n    >>> c()\n    2\n    '

    def c():
        if False:
            print('Hello World!')
        return x
    return c

def generator_func():
    if False:
        i = 10
        return i + 15
    '\n    >>> for i in generator_func(): print(i)\n    1\n    2\n    '
    yield 1
    yield 2