class FooError(ValueError):
    pass

def foo(s):
    if False:
        while True:
            i = 10
    n = int(s)
    if n == 0:
        raise FooError('invalid value: %s' % s)
    return 10 / n
foo('0')