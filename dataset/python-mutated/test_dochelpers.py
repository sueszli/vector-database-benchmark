"""
Tests for dochelpers.py
"""
import os
import sys
import pytest
from spyder_kernels.utils.dochelpers import getargtxt, getargspecfromtext, getdoc, getobj, getsignaturefromtext, isdefined

class Test(object):

    def method(self, x, y=2):
        if False:
            return 10
        pass

@pytest.mark.skipif(os.name == 'nt', reason='Only works on Linux and Mac')
@pytest.mark.skipif(sys.platform == 'darwin' and sys.version_info[:2] == (3, 8), reason='Fails on Mac with Python 3.8')
def test_dochelpers():
    if False:
        for i in range(10):
            print('nop')
    'Test dochelpers.'
    assert getargtxt(Test.method) == ['x, ', 'y=2']
    assert not getargtxt(Test.__init__)
    assert getdoc(sorted) == {'note': 'Function of builtins module', 'argspec': '(...)', 'docstring': 'Return a new list containing all items from the iterable in ascending order.\n\nA custom key function can be supplied to customize the sort order, and the\nreverse flag can be set to request the result in descending order.', 'name': 'sorted'}
    assert not getargtxt(sorted)
    assert isdefined('numpy.take', force_import=True)
    assert isdefined('__import__')
    assert not isdefined('zzz', force_import=True)
    assert getobj('globals') == 'globals'
    assert not getobj('globals().keys')
    assert getobj('+scipy.signal.') == 'scipy.signal'
    assert getobj('4.') == '4'

def test_no_signature():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test that we can get documentation for objects for which Python can't get a\n    signature directly because it gives an error.\n\n    This is a regression test for issue spyder-ide/spyder#21148\n    "
    import numpy as np
    doc = getdoc(np.where)
    signature = doc['argspec']
    assert signature and signature != '(...)' and signature.startswith('(')
    assert doc['docstring']

@pytest.mark.parametrize('text, name, expected', [('foo(x, y)', 'foo', '(x, y)'), ('foo(x, y)', '', '(x, y)'), ('foo(x)', '', '(x)'), ('foo(x = {})', '', '(x = {})'), ('1a(x, y)', '', ''), ('a1(x, y=2)', '', '(x, y=2)'), ('ΣΔ(x, y)', 'ΣΔ', '(x, y)'), ('ΣΔ(x, y)', '', '(x, y)'), ('ΣΔ(x, y) foo(a, b)', '', '(x, y)'), ('1a(x, y) foo(a, b)', '', '(a, b)'), ('foo(a, b = 1)\n\nΣΔ(x, y=2)', '', '(a, b = 1)'), ('1a(a, b = 1)\n\nΣΔ(x, y=2)', '', '(x, y=2)'), ('2(3 + 5) 3*(99) ΣΔ(x, y)', '', '(x, y)'), ('(x, y)', '', ''), ('foo (a=1, b = 2)', '', ''), ('foo()', '', ''), ('foo()', 'foo', '')])
def test_getsignaturefromtext(text, name, expected):
    if False:
        i = 10
        return i + 15
    assert getsignaturefromtext(text, name) == expected

def test_multisignature():
    if False:
        while True:
            i = 10
    '\n    Test that we can get at least one signature from an object with multiple\n    ones declared in its docstring.\n    '

    def foo():
        if False:
            i = 10
            return i + 15
        '\n        foo(x, y) foo(a, b)\n        foo(c, d)\n        '
    signature = getargspecfromtext(foo.__doc__)
    assert signature == '(x, y)'

def test_multiline_signature():
    if False:
        return 10
    '\n    Test that we can get signatures splitted into multiple lines in a\n    docstring.\n    '

    def foo():
        if False:
            print('Hello World!')
        '\n        foo(x,\n            y)\n\n        This is a docstring.\n        '
    signature = getargspecfromtext(foo.__doc__)
    assert signature.startswith('(x, ')
if __name__ == '__main__':
    pytest.main()