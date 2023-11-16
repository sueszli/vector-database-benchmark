import cython

def foo(egg):
    if False:
        while True:
            i = 10
    if not cython.compiled:
        egg = float(egg)
    return egg

def foo_defval(egg=1):
    if False:
        while True:
            i = 10
    if not cython.compiled:
        egg = float(egg)
    return egg ** 2

def cpfoo(egg=False):
    if False:
        for i in range(10):
            print('nop')
    if not cython.compiled:
        egg = bool(egg)
        v = int(not egg)
    else:
        v = not egg
    return (egg, v)

def test_pxd_locals():
    if False:
        while True:
            i = 10
    '\n    >>> v1, v2, v3 = test_pxd_locals()\n    >>> isinstance(v1, float)\n    True\n    >>> isinstance(v2, float)\n    True\n    >>> v3\n    (True, 0)\n    '
    return (foo(1), foo_defval(), cpfoo(1))