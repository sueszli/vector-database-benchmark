"""A module to test whether doctest recognizes some 2.2 features,
like static and class methods.

>>> print('yup')  # 1
yup

We include some (random) encoded (utf-8) text in the text surrounding
the example.  It should be ignored:

ЉЊЈЁЂ

"""
import sys
import unittest
if sys.flags.optimize >= 2:
    raise unittest.SkipTest('Cannot test docstrings with -O2')

class C(object):
    """Class C.

    >>> print(C())  # 2
    42


    We include some (random) encoded (utf-8) text in the text surrounding
    the example.  It should be ignored:

        ЉЊЈЁЂ

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'C.__init__.\n\n        >>> print(C()) # 3\n        42\n        '

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        >>> print(C()) # 4\n        42\n        '
        return '42'

    class D(object):
        """A nested D class.

        >>> print("In D!")   # 5
        In D!
        """

        def nested(self):
            if False:
                i = 10
                return i + 15
            '\n            >>> print(3) # 6\n            3\n            '

    def getx(self):
        if False:
            while True:
                i = 10
        '\n        >>> c = C()    # 7\n        >>> c.x = 12   # 8\n        >>> print(c.x)  # 9\n        -12\n        '
        return -self._x

    def setx(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        >>> c = C()     # 10\n        >>> c.x = 12    # 11\n        >>> print(c.x)   # 12\n        -12\n        '
        self._x = value
    x = property(getx, setx, doc='        >>> c = C()    # 13\n        >>> c.x = 12   # 14\n        >>> print(c.x)  # 15\n        -12\n        ')

    @staticmethod
    def statm():
        if False:
            for i in range(10):
                print('nop')
        '\n        A static method.\n\n        >>> print(C.statm())    # 16\n        666\n        >>> print(C().statm())  # 17\n        666\n        '
        return 666

    @classmethod
    def clsm(cls, val):
        if False:
            return 10
        '\n        A class method.\n\n        >>> print(C.clsm(22))    # 18\n        22\n        >>> print(C().clsm(23))  # 19\n        23\n        '
        return val

class Test(unittest.TestCase):

    def test_testmod(self):
        if False:
            print('Hello World!')
        import doctest, sys
        EXPECTED = 19
        (f, t) = doctest.testmod(sys.modules[__name__])
        if f:
            self.fail('%d of %d doctests failed' % (f, t))
        if t != EXPECTED:
            self.fail('expected %d tests to run, not %d' % (EXPECTED, t))
from doctest import *
if __name__ == '__main__':
    unittest.main()