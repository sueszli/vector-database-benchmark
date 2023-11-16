"""This is a sample module that doesn't really test anything all that
   interesting.

It simply has a few tests, some of which succeed and some of which fail.

It's important that the numbers remain constant as another test is
testing the running of these tests.


>>> 2+2
4
"""

def foo():
    if False:
        i = 10
        return i + 15
    '\n\n    >>> 2+2\n    5\n\n    >>> 2+2\n    4\n    '

def bar():
    if False:
        while True:
            i = 10
    '\n\n    >>> 2+2\n    4\n    '

def test_silly_setup():
    if False:
        return 10
    '\n\n    >>> import test.test_doctest\n    >>> test.test_doctest.sillySetup\n    True\n    '

def w_blank():
    if False:
        return 10
    "\n    >>> if 1:\n    ...    print('a')\n    ...    print()\n    ...    print('b')\n    a\n    <BLANKLINE>\n    b\n    "
x = 1

def x_is_one():
    if False:
        print('Hello World!')
    '\n    >>> x\n    1\n    '

def y_is_one():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> y\n    1\n    '
__test__ = {'good': '\n                    >>> 42\n                    42\n                    ', 'bad': '\n                    >>> 42\n                    666\n                    '}

def test_suite():
    if False:
        print('Hello World!')
    import doctest
    return doctest.DocTestSuite()