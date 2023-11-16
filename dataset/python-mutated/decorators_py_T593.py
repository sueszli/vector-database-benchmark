"""
>>> am_i_buggy
False
"""

def testme(func):
    if False:
        print('Hello World!')
    try:
        am_i_buggy
        return True
    except NameError:
        return False

@testme
def am_i_buggy():
    if False:
        for i in range(10):
            print('nop')
    pass

def called_deco(a, b, c):
    if False:
        return 10
    a.append((1, b, c))

    def count(f):
        if False:
            return 10
        a.append((2, b, c))
        return f
    return count
L = []

@called_deco(L, 5, c=6)
@called_deco(L, c=3, b=4)
@called_deco(L, 1, 2)
def wrapped_func(x):
    if False:
        print('Hello World!')
    '\n    >>> L\n    [(1, 5, 6), (1, 4, 3), (1, 1, 2), (2, 1, 2), (2, 4, 3), (2, 5, 6)]\n    >>> wrapped_func(99)\n    99\n    >>> L\n    [(1, 5, 6), (1, 4, 3), (1, 1, 2), (2, 1, 2), (2, 4, 3), (2, 5, 6)]\n    '
    return x

def class_in_closure(x):
    if False:
        print('Hello World!')
    '\n    >>> C1, c0 = class_in_closure(5)\n    >>> C1().smeth1()\n    (5, ())\n    >>> C1.smeth1(1,2)\n    (5, (1, 2))\n    >>> C1.smeth1()\n    (5, ())\n    >>> c0.smeth0()\n    1\n    >>> c0.__class__.smeth0()\n    1\n    '

    class ClosureClass1(object):

        @staticmethod
        def smeth1(*args):
            if False:
                i = 10
                return i + 15
            return (x, args)

    class ClosureClass0(object):

        @staticmethod
        def smeth0():
            if False:
                print('Hello World!')
            return 1
    return (ClosureClass1, ClosureClass0())

def class_not_in_closure():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> c = class_not_in_closure()\n    >>> c.smeth0()\n    1\n    >>> c.__class__.smeth0()\n    1\n    '

    class ClosureClass0(object):

        @staticmethod
        def smeth0():
            if False:
                while True:
                    i = 10
            return 1
    return ClosureClass0()