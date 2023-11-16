""" Doctests for NumPy-specific nose/doctest modifications

"""
from __future__ import division, absolute_import, print_function

def check_random_directive():
    if False:
        while True:
            i = 10
    '\n    >>> 2+2\n    <BadExample object at 0x084D05AC>  #random: may vary on your system\n    '

def check_implicit_np():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> np.array([1,2,3])\n    array([1, 2, 3])\n    '

def check_whitespace_enabled():
    if False:
        while True:
            i = 10
    '\n    # whitespace after the 3\n    >>> 1+2\n    3\n\n    # whitespace before the 7\n    >>> 3+4\n     7\n    '

def check_empty_output():
    if False:
        print('Hello World!')
    ' Check that no output does not cause an error.\n\n    This is related to nose bug 445; the numpy plugin changed the\n    doctest-result-variable default and therefore hit this bug:\n    http://code.google.com/p/python-nose/issues/detail?id=445\n\n    >>> a = 10\n    '

def check_skip():
    if False:
        for i in range(10):
            print('nop')
    ' Check skip directive\n\n    The test below should not run\n\n    >>> 1/0 #doctest: +SKIP\n    '
if __name__ == '__main__':
    import nose
    from numpy.testing.noseclasses import NumpyDoctest
    argv = ['', __file__, '--with-numpydoctest']
    nose.core.TestProgram(argv=argv, addplugins=[NumpyDoctest()])