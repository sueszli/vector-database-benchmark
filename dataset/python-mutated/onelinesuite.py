"""
>>> y  # doctest: +ELLIPSIS
Traceback (most recent call last):
NameError: ...name 'y' is not defined
>>> z  # doctest: +ELLIPSIS
Traceback (most recent call last):
NameError: ...name 'z' is not defined
>>> f()
17
"""
x = False
if x:
    y = 42
    z = 88

def f():
    if False:
        for i in range(10):
            print('nop')
    return 17

def suite_in_func(x):
    if False:
        i = 10
        return i + 15
    '\n    >>> suite_in_func(True)\n    (42, 88)\n    >>> suite_in_func(False)\n    (0, 0)\n    '
    y = z = 0
    if x:
        y = 42
        z = 88
    return (y, z)