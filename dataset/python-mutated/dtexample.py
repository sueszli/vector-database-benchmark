"""Simple example using doctests.

This file just contains doctests both using plain python and IPython prompts.
All tests should be loaded by nose.
"""
import os

def pyfunc():
    if False:
        for i in range(10):
            print('nop')
    "Some pure python tests...\n\n    >>> pyfunc()\n    'pyfunc'\n\n    >>> import os\n\n    >>> 2+3\n    5\n\n    >>> for i in range(3):\n    ...     print(i, end=' ')\n    ...     print(i+1, end=' ')\n    ...\n    0 1 1 2 2 3 \n    "
    return 'pyfunc'

def ipfunc():
    if False:
        print('Hello World!')
    "Some ipython tests...\n\n    In [1]: import os\n\n    In [3]: 2+3\n    Out[3]: 5\n\n    In [26]: for i in range(3):\n       ....:     print(i, end=' ')\n       ....:     print(i+1, end=' ')\n       ....:\n    0 1 1 2 2 3\n\n\n    It's OK to use '_' for the last result, but do NOT try to use IPython's\n    numbered history of _NN outputs, since those won't exist under the\n    doctest environment:\n\n    In [7]: 'hi'\n    Out[7]: 'hi'\n\n    In [8]: print(repr(_))\n    'hi'\n\n    In [7]: 3+4\n    Out[7]: 7\n\n    In [8]: _+3\n    Out[8]: 10\n\n    In [9]: ipfunc()\n    Out[9]: 'ipfunc'\n    "
    return 'ipfunc'

def ipos():
    if False:
        i = 10
        return i + 15
    'Examples that access the operating system work:\n\n    In [1]: !echo hello\n    hello\n\n    In [2]: !echo hello > /tmp/foo_iptest\n\n    In [3]: !cat /tmp/foo_iptest\n    hello\n\n    In [4]: rm -f /tmp/foo_iptest\n    '
    pass
ipos.__skip_doctest__ = os.name == 'nt'

def ranfunc():
    if False:
        i = 10
        return i + 15
    "A function with some random output.\n\n       Normal examples are verified as usual:\n       >>> 1+3\n       4\n\n       But if you put '# random' in the output, it is ignored:\n       >>> 1+3\n       junk goes here...  # random\n\n       >>> 1+2\n       again,  anything goes #random\n       if multiline, the random mark is only needed once.\n\n       >>> 1+2\n       You can also put the random marker at the end:\n       # random\n\n       >>> 1+2\n       # random\n       .. or at the beginning.\n\n       More correct input is properly verified:\n       >>> ranfunc()\n       'ranfunc'\n    "
    return 'ranfunc'

def random_all():
    if False:
        print('Hello World!')
    'A function where we ignore the output of ALL examples.\n\n    Examples:\n\n      # all-random\n\n      This mark tells the testing machinery that all subsequent examples should\n      be treated as random (ignoring their output).  They are still executed,\n      so if a they raise an error, it will be detected as such, but their\n      output is completely ignored.\n\n      >>> 1+3\n      junk goes here...\n\n      >>> 1+3\n      klasdfj;\n\n      >>> 1+2\n      again,  anything goes\n      blah...\n    '
    pass

def iprand():
    if False:
        print('Hello World!')
    "Some ipython tests with random output.\n\n    In [7]: 3+4\n    Out[7]: 7\n\n    In [8]: print('hello')\n    world  # random\n\n    In [9]: iprand()\n    Out[9]: 'iprand'\n    "
    return 'iprand'

def iprand_all():
    if False:
        while True:
            i = 10
    "Some ipython tests with fully random output.\n\n    # all-random\n    \n    In [7]: 1\n    Out[7]: 99\n\n    In [8]: print('hello')\n    world\n\n    In [9]: iprand_all()\n    Out[9]: 'junk'\n    "
    return 'iprand_all'