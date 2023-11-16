"""Tests for the key interactiveshell module, where the main ipython class is defined.
"""
import stack_data
import sys
SV_VERSION = tuple([int(x) for x in stack_data.__version__.split('.')[0:2]])

def test_reset():
    if False:
        return 10
    'reset must clear most namespaces.'
    ip.reset()
    nvars_user_ns = len(ip.user_ns)
    nvars_hidden = len(ip.user_ns_hidden)
    ip.user_ns['x'] = 1
    ip.user_ns['y'] = 1
    ip.reset()
    assert len(ip.user_ns) == nvars_user_ns
    assert len(ip.user_ns_hidden) == nvars_hidden

def doctest_tb_plain():
    if False:
        return 10
    '\n    In [18]: xmode plain\n    Exception reporting mode: Plain\n\n    In [19]: run simpleerr.py\n    Traceback (most recent call last):\n      File ...:...\n        bar(mode)\n      File ...:... in bar\n        div0()\n      File ...:... in div0\n        x/y\n    ZeroDivisionError: ...\n    '

def doctest_tb_context():
    if False:
        while True:
            i = 10
    '\n    In [3]: xmode context\n    Exception reporting mode: Context\n\n    In [4]: run simpleerr.py\n    ---------------------------------------------------------------------------\n    ZeroDivisionError                         Traceback (most recent call last)\n    <BLANKLINE>\n    ...\n         30     except IndexError:\n         31         mode = \'div\'\n    ---> 33     bar(mode)\n    <BLANKLINE>\n    ... in bar(mode)\n         15     "bar"\n         16     if mode==\'div\':\n    ---> 17         div0()\n         18     elif mode==\'exit\':\n         19         try:\n    <BLANKLINE>\n    ... in div0()\n          6     x = 1\n          7     y = 0\n    ----> 8     x/y\n    <BLANKLINE>\n    ZeroDivisionError: ...'

def doctest_tb_verbose():
    if False:
        while True:
            i = 10
    '\n    In [5]: xmode verbose\n    Exception reporting mode: Verbose\n\n    In [6]: run simpleerr.py\n    ---------------------------------------------------------------------------\n    ZeroDivisionError                         Traceback (most recent call last)\n    <BLANKLINE>\n    ...\n         30     except IndexError:\n         31         mode = \'div\'\n    ---> 33     bar(mode)\n            mode = \'div\'\n    <BLANKLINE>\n    ... in bar(mode=\'div\')\n         15     "bar"\n         16     if mode==\'div\':\n    ---> 17         div0()\n         18     elif mode==\'exit\':\n         19         try:\n    <BLANKLINE>\n    ... in div0()\n          6     x = 1\n          7     y = 0\n    ----> 8     x/y\n            x = 1\n            y = 0\n    <BLANKLINE>\n    ZeroDivisionError: ...\n    '

def doctest_tb_sysexit():
    if False:
        for i in range(10):
            print('nop')
    '\n    In [17]: %xmode plain\n    Exception reporting mode: Plain\n\n    In [18]: %run simpleerr.py exit\n    An exception has occurred, use %tb to see the full traceback.\n    SystemExit: (1, \'Mode = exit\')\n\n    In [19]: %run simpleerr.py exit 2\n    An exception has occurred, use %tb to see the full traceback.\n    SystemExit: (2, \'Mode = exit\')\n\n    In [20]: %tb\n    Traceback (most recent call last):\n      File ...:... in execfile\n        exec(compiler(f.read(), fname, "exec"), glob, loc)\n      File ...:...\n        bar(mode)\n      File ...:... in bar\n        sysexit(stat, mode)\n      File ...:... in sysexit\n        raise SystemExit(stat, f"Mode = {mode}")\n    SystemExit: (2, \'Mode = exit\')\n\n    In [21]: %xmode context\n    Exception reporting mode: Context\n\n    In [22]: %tb\n    ---------------------------------------------------------------------------\n    SystemExit                                Traceback (most recent call last)\n    File ..., in execfile(fname, glob, loc, compiler)\n         ... with open(fname, "rb") as f:\n         ...     compiler = compiler or compile\n    ---> ...     exec(compiler(f.read(), fname, "exec"), glob, loc)\n    ...\n         30     except IndexError:\n         31         mode = \'div\'\n    ---> 33     bar(mode)\n    <BLANKLINE>\n    ...bar(mode)\n         21         except:\n         22             stat = 1\n    ---> 23         sysexit(stat, mode)\n         24     else:\n         25         raise ValueError(\'Unknown mode\')\n    <BLANKLINE>\n    ...sysexit(stat, mode)\n         10 def sysexit(stat, mode):\n    ---> 11     raise SystemExit(stat, f"Mode = {mode}")\n    <BLANKLINE>\n    SystemExit: (2, \'Mode = exit\')\n    '
if SV_VERSION < (0, 6):

    def doctest_tb_sysexit_verbose_stack_data_05():
        if False:
            for i in range(10):
                print('nop')
        '\n        In [18]: %run simpleerr.py exit\n        An exception has occurred, use %tb to see the full traceback.\n        SystemExit: (1, \'Mode = exit\')\n\n        In [19]: %run simpleerr.py exit 2\n        An exception has occurred, use %tb to see the full traceback.\n        SystemExit: (2, \'Mode = exit\')\n\n        In [23]: %xmode verbose\n        Exception reporting mode: Verbose\n\n        In [24]: %tb\n        ---------------------------------------------------------------------------\n        SystemExit                                Traceback (most recent call last)\n        <BLANKLINE>\n        ...\n            30     except IndexError:\n            31         mode = \'div\'\n        ---> 33     bar(mode)\n                mode = \'exit\'\n        <BLANKLINE>\n        ... in bar(mode=\'exit\')\n            ...     except:\n            ...         stat = 1\n        ---> ...     sysexit(stat, mode)\n                mode = \'exit\'\n                stat = 2\n            ...     else:\n            ...         raise ValueError(\'Unknown mode\')\n        <BLANKLINE>\n        ... in sysexit(stat=2, mode=\'exit\')\n            10 def sysexit(stat, mode):\n        ---> 11     raise SystemExit(stat, f"Mode = {mode}")\n                stat = 2\n        <BLANKLINE>\n        SystemExit: (2, \'Mode = exit\')\n        '

def test_run_cell():
    if False:
        return 10
    import textwrap
    ip.run_cell('a = 10\na+=1')
    ip.run_cell('assert a == 11\nassert 1')
    assert ip.user_ns['a'] == 11
    complex = textwrap.dedent('\n    if 1:\n        print "hello"\n        if 1:\n            print "world"\n        \n    if 2:\n        print "foo"\n\n    if 3:\n        print "bar"\n\n    if 4:\n        print "bar"\n    \n    ')
    ip.run_cell(complex)

def test_db():
    if False:
        for i in range(10):
            print('nop')
    'Test the internal database used for variable persistence.'
    ip.db['__unittest_'] = 12
    assert ip.db['__unittest_'] == 12
    del ip.db['__unittest_']
    assert '__unittest_' not in ip.db