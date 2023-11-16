def test_inner(a):
    if False:
        print('Hello World!')
    '\n    >>> a = test_inner(1)\n    >>> b = test_inner(2)\n    >>> a()\n    1\n    >>> b()\n    2\n    '

    def inner(b=a):
        if False:
            return 10
        return b
    return inner

def test_lambda(n):
    if False:
        print('Hello World!')
    '\n    >>> [f() for f in test_lambda(3)]\n    [0, 1, 2]\n    '
    return [lambda v=i: v for i in range(n)]