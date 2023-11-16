"""Tests for the ipdoctest machinery itself.

Note: in a file named test_X, functions whose only test is their docstring (as
a doctest) and which have no test functionality of their own, should be called
'doctest_foo' instead of 'test_foo', otherwise they get double-counted (the
empty function call is counted as a test, which just inflates tests numbers
artificially).
"""

def doctest_simple():
    if False:
        for i in range(10):
            print('nop')
    'ipdoctest must handle simple inputs\n    \n    In [1]: 1\n    Out[1]: 1\n\n    In [2]: print(1)\n    1\n    '

def doctest_multiline1():
    if False:
        i = 10
        return i + 15
    'The ipdoctest machinery must handle multiline examples gracefully.\n\n    In [2]: for i in range(4):\n       ...:     print(i)\n       ...:      \n    0\n    1\n    2\n    3\n    '

def doctest_multiline2():
    if False:
        print('Hello World!')
    "Multiline examples that define functions and print output.\n\n    In [7]: def f(x):\n       ...:     return x+1\n       ...: \n\n    In [8]: f(1)\n    Out[8]: 2\n\n    In [9]: def g(x):\n       ...:     print('x is:',x)\n       ...:      \n\n    In [10]: g(1)\n    x is: 1\n\n    In [11]: g('hello')\n    x is: hello\n    "

def doctest_multiline3():
    if False:
        return 10
    'Multiline examples with blank lines.\n\n    In [12]: def h(x):\n       ....:     if x>1:\n       ....:         return x**2\n       ....:     # To leave a blank line in the input, you must mark it\n       ....:     # with a comment character:\n       ....:     #\n       ....:     # otherwise the doctest parser gets confused.\n       ....:     else:\n       ....:         return -1\n       ....:      \n\n    In [13]: h(5)\n    Out[13]: 25\n\n    In [14]: h(1)\n    Out[14]: -1\n\n    In [15]: h(0)\n    Out[15]: -1\n   '

def doctest_builtin_underscore():
    if False:
        i = 10
        return i + 15
    'Defining builtins._ should not break anything outside the doctest\n    while also should be working as expected inside the doctest.\n\n    In [1]: import builtins\n\n    In [2]: builtins._ = 42\n\n    In [3]: builtins._\n    Out[3]: 42\n\n    In [4]: _\n    Out[4]: 42\n    '