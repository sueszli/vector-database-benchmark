"""These kinds of tests are less than ideal, but at least they run.

This was an old test that was being run interactively in the top-level tests/
directory, which we are removing.  For now putting this here ensures at least
we do run the test, though ultimately this functionality should all be tested
with better-isolated tests that don't rely on the global instance in iptest.
"""
from IPython.core.splitinput import LineInfo
from IPython.core.prefilter import AutocallChecker

def doctest_autocall():
    if False:
        for i in range(10):
            print('nop')
    '\n    In [1]: def f1(a,b,c):\n       ...:     return a+b+c\n       ...:\n\n    In [2]: def f2(a):\n       ...:     return a + a\n       ...:\n\n    In [3]: def r(x):\n       ...:     return True\n       ...:\n\n    In [4]: ;f2 a b c\n    Out[4]: \'a b ca b c\'\n\n    In [5]: assert _ == "a b ca b c"\n\n    In [6]: ,f1 a b c\n    Out[6]: \'abc\'\n\n    In [7]: assert _ == \'abc\'\n\n    In [8]: print(_)\n    abc\n\n    In [9]: /f1 1,2,3\n    Out[9]: 6\n\n    In [10]: assert _ == 6\n\n    In [11]: /f2 4\n    Out[11]: 8\n\n    In [12]: assert _ == 8\n\n    In [12]: del f1, f2\n\n    In [13]: ,r a\n    Out[13]: True\n\n    In [14]: assert _ == True\n\n    In [15]: r\'a\'\n    Out[15]: \'a\'\n\n    In [16]: assert _ == \'a\'\n    '

def test_autocall_should_ignore_raw_strings():
    if False:
        print('Hello World!')
    line_info = LineInfo("r'a'")
    pm = ip.prefilter_manager
    ac = AutocallChecker(shell=pm.shell, prefilter_manager=pm, config=pm.config)
    assert ac.check(line_info) is None