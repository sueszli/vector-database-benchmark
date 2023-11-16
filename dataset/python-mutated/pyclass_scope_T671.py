A = 1234

class SimpleAssignment(object):
    """
    >>> SimpleAssignment.A
    1234
    """
    A = A

class SimpleRewrite(object):
    """
    >>> SimpleRewrite.A
    4321
    """
    A = 4321
    A = A

def simple_inner(a):
    if False:
        while True:
            i = 10
    '\n    >>> simple_inner(4321).A\n    1234\n    '
    A = a

    class X(object):
        A = A
    return X

def conditional(a, cond):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> conditional(4321, False).A\n    1234\n    >>> conditional(4321, True).A\n    4321\n    '

    class X(object):
        if cond:
            A = a
        A = A
    return X

def name_error():
    if False:
        print('Hello World!')
    '\n    >>> name_error() #doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    NameError: ...B...\n    '

    class X(object):
        B = B

def conditional_name_error(cond):
    if False:
        i = 10
        return i + 15
    '\n    >>> conditional_name_error(True).B\n    4321\n    >>> conditional_name_error(False).B #doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    NameError: ...B...\n    '

    class X(object):
        if cond:
            B = 4321
        B = B
    return X
C = 1111
del C

def name_error_deleted():
    if False:
        return 10
    '\n    >>> name_error_deleted() #doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    NameError: ...C...\n    '

    class X(object):
        C = C
_set = set

def name_lookup_order():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> Scope = name_lookup_order()\n    >>> Scope().set(2)\n    42\n    >>> Scope.test1 == _set()\n    True\n    >>> Scope.test2 == _set()\n    True\n\n    '

    class Scope(object):
        test1 = set()
        test2 = set()

        def set(self, x):
            if False:
                return 10
            return 42
    return Scope