def foo1(bar, baz=1):
    if False:
        while True:
            i = 10
    return 1

def foo2(bar, baz, qux=1):
    if False:
        i = 10
        return i + 15
    return 2

def foo3(bar, baz=1, qux=2):
    if False:
        return 10
    return 3

def foo4(bar, baz, qux=1, quux=2):
    if False:
        while True:
            i = 10
    return 4

def _walk_dir(dir, ddir=None, maxlevels=10, quiet=0):
    if False:
        for i in range(10):
            print('nop')
    return