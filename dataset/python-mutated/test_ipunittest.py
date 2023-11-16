"""Tests for IPython's test support utilities.

These are decorators that allow standalone functions and docstrings to be seen
as tests by unittest, replicating some of nose's functionality.  Additionally,
IPython-syntax docstrings can be auto-converted to '>>>' so that ipython
sessions can be copy-pasted as tests.

This file can be run as a script, and it will call unittest.main().  We must
check that it works with unittest as well as with nose...


Notes:

- Using nosetests --with-doctest --doctest-tests testfile.py
  will find docstrings as tests wherever they are, even in methods.  But
  if we use ipython syntax in the docstrings, they must be decorated with
  @ipdocstring.  This is OK for test-only code, but not for user-facing
  docstrings where we want to keep the ipython syntax.

- Using nosetests --with-doctest file.py
  also finds doctests if the file name doesn't have 'test' in it, because it is
  treated like a normal module.  But if nose treats the file like a test file,
  then for normal classes to be doctested the extra --doctest-tests is
  necessary.

- running this script with python (it has a __main__ section at the end) misses
  one docstring test, the one embedded in the Foo object method.  Since our
  approach relies on using decorators that create standalone TestCase
  instances, it can only be used for functions, not for methods of objects.
Authors
-------

- Fernando Perez <Fernando.Perez@berkeley.edu>
"""
from IPython.testing.ipunittest import ipdoctest, ipdocstring

@ipdoctest
def simple_dt():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> print(1+1)\n    2\n    '

@ipdoctest
def ipdt_flush():
    if False:
        while True:
            i = 10
    '\nIn [20]: print(1)\n1\n\nIn [26]: for i in range(4):\n   ....:     print(i)\n   ....:     \n   ....: \n0\n1\n2\n3\n\nIn [27]: 3+4\nOut[27]: 7\n'

@ipdoctest
def ipdt_indented_test():
    if False:
        i = 10
        return i + 15
    '\n    In [20]: print(1)\n    1\n\n    In [26]: for i in range(4):\n       ....:     print(i)\n       ....:     \n       ....: \n    0\n    1\n    2\n    3\n\n    In [27]: 3+4\n    Out[27]: 7\n    '

class Foo(object):
    """For methods, the normal decorator doesn't work.

    But rewriting the docstring with ip2py does, *but only if using nose
    --with-doctest*.  Do we want to have that as a dependency?
    """

    @ipdocstring
    def ipdt_method(self):
        if False:
            print('Hello World!')
        '\n        In [20]: print(1)\n        1\n\n        In [26]: for i in range(4):\n           ....:     print(i)\n           ....:     \n           ....: \n        0\n        1\n        2\n        3\n\n        In [27]: 3+4\n        Out[27]: 7\n        '

    def normaldt_method(self):
        if False:
            print('Hello World!')
        '\n        >>> print(1+1)\n        2\n        '