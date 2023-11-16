"""Tests for various magic functions."""
import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class, register_cell_magic, register_line_magic
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile

@magic.magics_class
class DummyMagics(magic.Magics):
    pass

def test_extract_code_ranges():
    if False:
        while True:
            i = 10
    instr = '1 3 5-6 7-9 10:15 17: :10 10- -13 :'
    expected = [(0, 1), (2, 3), (4, 6), (6, 9), (9, 14), (16, None), (None, 9), (9, None), (None, 13), (None, None)]
    actual = list(code.extract_code_ranges(instr))
    assert actual == expected

def test_extract_symbols():
    if False:
        while True:
            i = 10
    source = 'import foo\na = 10\ndef b():\n    return 42\n\n\nclass A: pass\n\n\n'
    symbols_args = ['a', 'b', 'A', 'A,b', 'A,a', 'z']
    expected = [([], ['a']), (['def b():\n    return 42\n'], []), (['class A: pass\n'], []), (['class A: pass\n', 'def b():\n    return 42\n'], []), (['class A: pass\n'], ['a']), ([], ['z'])]
    for (symbols, exp) in zip(symbols_args, expected):
        assert code.extract_symbols(source, symbols) == exp

def test_extract_symbols_raises_exception_with_non_python_code():
    if False:
        print('Hello World!')
    source = "=begin A Ruby program :)=end\ndef hello\nputs 'Hello world'\nend"
    with pytest.raises(SyntaxError):
        code.extract_symbols(source, 'hello')

def test_magic_not_found():
    if False:
        print('Hello World!')
    with pytest.raises(UsageError):
        _ip.run_line_magic('doesntexist', '')
    result = _ip.run_cell('%doesntexist')
    assert isinstance(result.error_in_exec, UsageError)

def test_cell_magic_not_found():
    if False:
        i = 10
        return i + 15
    with pytest.raises(UsageError):
        _ip.run_cell_magic('doesntexist', 'line', 'cell')
    result = _ip.run_cell('%%doesntexist')
    assert isinstance(result.error_in_exec, UsageError)

def test_magic_error_status():
    if False:
        print('Hello World!')

    def fail(shell):
        if False:
            for i in range(10):
                print('nop')
        1 / 0
    _ip.register_magic_function(fail)
    result = _ip.run_cell('%fail')
    assert isinstance(result.error_in_exec, ZeroDivisionError)

def test_config():
    if False:
        return 10
    ' test that config magic does not raise\n    can happen if Configurable init is moved too early into\n    Magics.__init__ as then a Config object will be registered as a\n    magic.\n    '
    _ip.run_line_magic('config', '')

def test_config_available_configs():
    if False:
        return 10
    ' test that config magic prints available configs in unique and\n    sorted order. '
    with capture_output() as captured:
        _ip.run_line_magic('config', '')
    stdout = captured.stdout
    config_classes = stdout.strip().split('\n')[1:]
    assert config_classes == sorted(set(config_classes))

def test_config_print_class():
    if False:
        for i in range(10):
            print('nop')
    " test that config with a classname prints the class's options. "
    with capture_output() as captured:
        _ip.run_line_magic('config', 'TerminalInteractiveShell')
    stdout = captured.stdout
    assert re.match('TerminalInteractiveShell.* options', stdout.splitlines()[0]), f"{stdout}\n\n1st line of stdout not like 'TerminalInteractiveShell.* options'"

def test_rehashx():
    if False:
        for i in range(10):
            print('nop')
    _ip.alias_manager.clear_aliases()
    del _ip.db['syscmdlist']
    _ip.run_line_magic('rehashx', '')
    assert len(_ip.alias_manager.aliases) > 10
    for (name, cmd) in _ip.alias_manager.aliases:
        assert '.' not in name
    scoms = _ip.db['syscmdlist']
    assert len(scoms) > 10

def test_magic_parse_options():
    if False:
        i = 10
        return i + 15
    "Test that we don't mangle paths when parsing magic options."
    ip = get_ipython()
    path = 'c:\\x'
    m = DummyMagics(ip)
    opts = m.parse_options('-f %s' % path, 'f:')[0]
    if os.name == 'posix':
        expected = 'c:x'
    else:
        expected = path
    assert opts['f'] == expected

def test_magic_parse_long_options():
    if False:
        print('Hello World!')
    'Magic.parse_options can handle --foo=bar long options'
    ip = get_ipython()
    m = DummyMagics(ip)
    (opts, _) = m.parse_options('--foo --bar=bubble', 'a', 'foo', 'bar=')
    assert 'foo' in opts
    assert 'bar' in opts
    assert opts['bar'] == 'bubble'

def doctest_hist_f():
    if False:
        return 10
    "Test %hist -f with temporary filename.\n\n    In [9]: import tempfile\n\n    In [10]: tfile = tempfile.mktemp('.py','tmp-ipython-')\n\n    In [11]: %hist -nl -f $tfile 3\n\n    In [13]: import os; os.unlink(tfile)\n    "

def doctest_hist_op():
    if False:
        for i in range(10):
            print('nop')
    "Test %hist -op\n\n    In [1]: class b(float):\n       ...:     pass\n       ...:\n\n    In [2]: class s(object):\n       ...:     def __str__(self):\n       ...:         return 's'\n       ...:\n\n    In [3]:\n\n    In [4]: class r(b):\n       ...:     def __repr__(self):\n       ...:         return 'r'\n       ...:\n\n    In [5]: class sr(s,r): pass\n       ...:\n\n    In [6]:\n\n    In [7]: bb=b()\n\n    In [8]: ss=s()\n\n    In [9]: rr=r()\n\n    In [10]: ssrr=sr()\n\n    In [11]: 4.5\n    Out[11]: 4.5\n\n    In [12]: str(ss)\n    Out[12]: 's'\n\n    In [13]:\n\n    In [14]: %hist -op\n    >>> class b:\n    ...     pass\n    ...\n    >>> class s(b):\n    ...     def __str__(self):\n    ...         return 's'\n    ...\n    >>>\n    >>> class r(b):\n    ...     def __repr__(self):\n    ...         return 'r'\n    ...\n    >>> class sr(s,r): pass\n    >>>\n    >>> bb=b()\n    >>> ss=s()\n    >>> rr=r()\n    >>> ssrr=sr()\n    >>> 4.5\n    4.5\n    >>> str(ss)\n    's'\n    >>>\n    "

def test_hist_pof():
    if False:
        return 10
    ip = get_ipython()
    ip.run_cell('1+2', store_history=True)
    with TemporaryDirectory() as td:
        tf = os.path.join(td, 'hist.py')
        ip.run_line_magic('history', '-pof %s' % tf)
        assert os.path.isfile(tf)

def test_macro():
    if False:
        print('Hello World!')
    ip = get_ipython()
    ip.history_manager.reset()
    cmds = ['a=1', 'def b():\n  return a**2', 'print(a,b())']
    for (i, cmd) in enumerate(cmds, start=1):
        ip.history_manager.store_inputs(i, cmd)
    ip.run_line_magic('macro', 'test 1-3')
    assert ip.user_ns['test'].value == '\n'.join(cmds) + '\n'
    assert 'test' in ip.run_line_magic('macro', '')

def test_macro_run():
    if False:
        while True:
            i = 10
    'Test that we can run a multi-line macro successfully.'
    ip = get_ipython()
    ip.history_manager.reset()
    cmds = ['a=10', 'a+=1', 'print(a)', '%macro test 2-3']
    for cmd in cmds:
        ip.run_cell(cmd, store_history=True)
    assert ip.user_ns['test'].value == 'a+=1\nprint(a)\n'
    with tt.AssertPrints('12'):
        ip.run_cell('test')
    with tt.AssertPrints('13'):
        ip.run_cell('test')

def test_magic_magic():
    if False:
        i = 10
        return i + 15
    'Test %magic'
    ip = get_ipython()
    with capture_output() as captured:
        ip.run_line_magic('magic', '')
    stdout = captured.stdout
    assert '%magic' in stdout
    assert 'IPython' in stdout
    assert 'Available' in stdout

@dec.skipif_not_numpy
def test_numpy_reset_array_undec():
    if False:
        i = 10
        return i + 15
    "Test '%reset array' functionality"
    _ip.ex('import numpy as np')
    _ip.ex('a = np.empty(2)')
    assert 'a' in _ip.user_ns
    _ip.run_line_magic('reset', '-f array')
    assert 'a' not in _ip.user_ns

def test_reset_out():
    if False:
        print('Hello World!')
    "Test '%reset out' magic"
    _ip.run_cell("parrot = 'dead'", store_history=True)
    _ip.run_cell('parrot', store_history=True)
    assert 'dead' in [_ip.user_ns[x] for x in ('_', '__', '___')]
    _ip.run_line_magic('reset', '-f out')
    assert 'dead' not in [_ip.user_ns[x] for x in ('_', '__', '___')]
    assert len(_ip.user_ns['Out']) == 0

def test_reset_in():
    if False:
        print('Hello World!')
    "Test '%reset in' magic"
    _ip.run_cell('parrot', store_history=True)
    assert 'parrot' in [_ip.user_ns[x] for x in ('_i', '_ii', '_iii')]
    _ip.run_line_magic('reset', '-f in')
    assert 'parrot' not in [_ip.user_ns[x] for x in ('_i', '_ii', '_iii')]
    assert len(set(_ip.user_ns['In'])) == 1

def test_reset_dhist():
    if False:
        for i in range(10):
            print('nop')
    "Test '%reset dhist' magic"
    _ip.run_cell('tmp = [d for d in _dh]')
    _ip.run_line_magic('cd', os.path.dirname(pytest.__file__))
    _ip.run_line_magic('cd', '-')
    assert len(_ip.user_ns['_dh']) > 0
    _ip.run_line_magic('reset', '-f dhist')
    assert len(_ip.user_ns['_dh']) == 0
    _ip.run_cell('_dh = [d for d in tmp]')

def test_reset_in_length():
    if False:
        return 10
    "Test that '%reset in' preserves In[] length"
    _ip.run_cell("print 'foo'")
    _ip.run_cell('reset -f in')
    assert len(_ip.user_ns['In']) == _ip.displayhook.prompt_count + 1

class TestResetErrors(TestCase):

    def test_reset_redefine(self):
        if False:
            for i in range(10):
                print('nop')

        @magics_class
        class KernelMagics(Magics):

            @line_magic
            def less(self, shell):
                if False:
                    print('Hello World!')
                pass
        _ip.register_magics(KernelMagics)
        with self.assertLogs() as cm:
            import logging as log_mod
            log = log_mod.getLogger()
            log.info('Nothing')
            _ip.run_cell('reset -f')
        assert len(cm.output) == 1
        for out in cm.output:
            assert 'Invalid alias' not in out

def test_tb_syntaxerror():
    if False:
        print('Hello World!')
    'test %tb after a SyntaxError'
    ip = get_ipython()
    ip.run_cell('for')
    save_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        ip.run_cell('%tb')
        out = sys.stdout.getvalue()
    finally:
        sys.stdout = save_stdout
    last_line = out.rstrip().splitlines()[-1].strip()
    assert last_line == 'SyntaxError: invalid syntax'

def test_time():
    if False:
        for i in range(10):
            print('nop')
    ip = get_ipython()
    with tt.AssertPrints('Wall time: '):
        ip.run_cell('%time None')
    ip.run_cell('def f(kmjy):\n    %time print (2*kmjy)')
    with tt.AssertPrints('Wall time: '):
        with tt.AssertPrints('hihi', suppress=False):
            ip.run_cell("f('hi')")

def test_time_no_output_with_semicolon():
    if False:
        i = 10
        return i + 15
    ip = get_ipython()
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456')
    with tt.AssertNotPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456;')
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456 # Comment')
    with tt.AssertNotPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456; # Comment')
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456 # ;Comment')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456\n\n\n')
    with tt.AssertNotPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456;\n\n\n')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456  # Comment\n\n\n')
    with tt.AssertNotPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456;  # Comment\n\n\n')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456  # ;Comment\n\n\n')

def test_time_last_not_expression():
    if False:
        while True:
            i = 10
    ip.run_cell('%%time\nvar_1 = 1\nvar_2 = 2\n')
    assert ip.user_ns['var_1'] == 1
    del ip.user_ns['var_1']
    assert ip.user_ns['var_2'] == 2
    del ip.user_ns['var_2']

@dec.skip_win32
def test_time2():
    if False:
        print('Hello World!')
    ip = get_ipython()
    with tt.AssertPrints('CPU times: user '):
        ip.run_cell('%time None')

def test_time3():
    if False:
        for i in range(10):
            print('nop')
    'Erroneous magic function calls, issue gh-3334'
    ip = get_ipython()
    ip.user_ns.pop('run', None)
    with tt.AssertNotPrints('not found', channel='stderr'):
        ip.run_cell('%%time\nrun = 0\nrun += 1')

def test_multiline_time():
    if False:
        i = 10
        return i + 15
    'Make sure last statement from time return a value.'
    ip = get_ipython()
    ip.user_ns.pop('run', None)
    ip.run_cell(dedent('        %%time\n        a = "ho"\n        b = "hey"\n        a+b\n        '))
    assert ip.user_ns_hidden['_'] == 'hohey'

def test_time_local_ns():
    if False:
        print('Hello World!')
    '\n    Test that local_ns is actually global_ns when running a cell magic\n    '
    ip = get_ipython()
    ip.run_cell('%%time\nmyvar = 1')
    assert ip.user_ns['myvar'] == 1
    del ip.user_ns['myvar']

def test_capture():
    if False:
        return 10
    ip = get_ipython()
    ip.run_cell('%%capture abc\n1+2')
    with tt.AssertPrints('True', suppress=False):
        ip.run_cell("'abc' in locals()")
    with tt.AssertPrints('True', suppress=False):
        ip.run_cell("'outputs' in dir(abc)")
    with tt.AssertPrints('3', suppress=False):
        ip.run_cell('abc.outputs[0]')
    ip.run_cell('%%capture abc\n7+8;')
    with tt.AssertPrints('False', suppress=False):
        ip.run_cell("'abc' in locals()")

def test_doctest_mode():
    if False:
        while True:
            i = 10
    'Toggle doctest_mode twice, it should be a no-op and run without error'
    _ip.run_line_magic('doctest_mode', '')
    _ip.run_line_magic('doctest_mode', '')

def test_parse_options():
    if False:
        while True:
            i = 10
    'Tests for basic options parsing in magics.'
    m = DummyMagics(_ip)
    assert m.parse_options('foo', '')[1] == 'foo'
    assert m.parse_options('foo', '')[1] == 'foo'

def test_parse_options_preserve_non_option_string():
    if False:
        while True:
            i = 10
    'Test to assert preservation of non-option part of magic-block, while parsing magic options.'
    m = DummyMagics(_ip)
    (opts, stmt) = m.parse_options(' -n1  -r 13 _ = 314 + foo', 'n:r:', preserve_non_opts=True)
    assert opts == {'n': '1', 'r': '13'}
    assert stmt == '_ = 314 + foo'

def test_run_magic_preserve_code_block():
    if False:
        return 10
    'Test to assert preservation of non-option part of magic-block, while running magic.'
    _ip.user_ns['spaces'] = []
    _ip.run_line_magic('timeit', "-n1 -r1 spaces.append([s.count(' ') for s in ['document']])")
    assert _ip.user_ns['spaces'] == [[0]]

def test_dirops():
    if False:
        while True:
            i = 10
    'Test various directory handling operations.'
    curpath = os.getcwd
    startdir = os.getcwd()
    ipdir = os.path.realpath(_ip.ipython_dir)
    try:
        _ip.run_line_magic('cd', '"%s"' % ipdir)
        assert curpath() == ipdir
        _ip.run_line_magic('cd', '-')
        assert curpath() == startdir
        _ip.run_line_magic('pushd', '"%s"' % ipdir)
        assert curpath() == ipdir
        _ip.run_line_magic('popd', '')
        assert curpath() == startdir
    finally:
        os.chdir(startdir)

def test_cd_force_quiet():
    if False:
        i = 10
        return i + 15
    'Test OSMagics.cd_force_quiet option'
    _ip.config.OSMagics.cd_force_quiet = True
    osmagics = osm.OSMagics(shell=_ip)
    startdir = os.getcwd()
    ipdir = os.path.realpath(_ip.ipython_dir)
    try:
        with tt.AssertNotPrints(ipdir):
            osmagics.cd('"%s"' % ipdir)
        with tt.AssertNotPrints(startdir):
            osmagics.cd('-')
    finally:
        os.chdir(startdir)

def test_xmode():
    if False:
        i = 10
        return i + 15
    xmode = _ip.InteractiveTB.mode
    for i in range(4):
        _ip.run_line_magic('xmode', '')
    assert _ip.InteractiveTB.mode == xmode

def test_reset_hard():
    if False:
        for i in range(10):
            print('nop')
    monitor = []

    class A(object):

        def __del__(self):
            if False:
                while True:
                    i = 10
            monitor.append(1)

        def __repr__(self):
            if False:
                return 10
            return '<A instance>'
    _ip.user_ns['a'] = A()
    _ip.run_cell('a')
    assert monitor == []
    _ip.run_line_magic('reset', '-f')
    assert monitor == [1]

class TestXdel(tt.TempFileMixin):

    def test_xdel(self):
        if False:
            while True:
                i = 10
        'Test that references from %run are cleared by xdel.'
        src = 'class A(object):\n    monitor = []\n    def __del__(self):\n        self.monitor.append(1)\na = A()\n'
        self.mktmp(src)
        _ip.run_line_magic('run', '%s' % self.fname)
        _ip.run_cell('a')
        monitor = _ip.user_ns['A'].monitor
        assert monitor == []
        _ip.run_line_magic('xdel', 'a')
        gc.collect(0)
        assert monitor == [1]

def doctest_who():
    if False:
        print('Hello World!')
    "doctest for %who\n\n    In [1]: %reset -sf\n\n    In [2]: alpha = 123\n\n    In [3]: beta = 'beta'\n\n    In [4]: %who int\n    alpha\n\n    In [5]: %who str\n    beta\n\n    In [6]: %whos\n    Variable   Type    Data/Info\n    ----------------------------\n    alpha      int     123\n    beta       str     beta\n\n    In [7]: %who_ls\n    Out[7]: ['alpha', 'beta']\n    "

def test_whos():
    if False:
        i = 10
        return i + 15
    'Check that whos is protected against objects where repr() fails.'

    class A(object):

        def __repr__(self):
            if False:
                print('Hello World!')
            raise Exception()
    _ip.user_ns['a'] = A()
    _ip.run_line_magic('whos', '')

def doctest_precision():
    if False:
        while True:
            i = 10
    "doctest for %precision\n\n    In [1]: f = get_ipython().display_formatter.formatters['text/plain']\n\n    In [2]: %precision 5\n    Out[2]: '%.5f'\n\n    In [3]: f.float_format\n    Out[3]: '%.5f'\n\n    In [4]: %precision %e\n    Out[4]: '%e'\n\n    In [5]: f(3.1415927)\n    Out[5]: '3.141593e+00'\n    "

def test_debug_magic():
    if False:
        i = 10
        return i + 15
    'Test debugging a small code with %debug\n\n    In [1]: with PdbTestInput([\'c\']):\n       ...:     %debug print("a b") #doctest: +ELLIPSIS\n       ...:\n    ...\n    ipdb> c\n    a b\n    In [2]:\n    '

def test_debug_magic_locals():
    if False:
        return 10
    "Test debugging a small code with %debug with locals\n\n    In [1]: with PdbTestInput(['c']):\n       ...:     def fun():\n       ...:         res = 1\n       ...:         %debug print(res)\n       ...:     fun()\n       ...:\n    ...\n    ipdb> c\n    1\n    In [2]:\n    "

def test_psearch():
    if False:
        while True:
            i = 10
    with tt.AssertPrints('dict.fromkeys'):
        _ip.run_cell('dict.fr*?')
    with tt.AssertPrints('π.is_integer'):
        _ip.run_cell('π = 3.14;\nπ.is_integ*?')

def test_timeit_shlex():
    if False:
        i = 10
        return i + 15
    'test shlex issues with timeit (#1109)'
    _ip.ex('def f(*a,**kw): pass')
    _ip.run_line_magic('timeit', '-n1 "this is a bug".count(" ")')
    _ip.run_line_magic('timeit', '-r1 -n1 f(" ", 1)')
    _ip.run_line_magic('timeit', '-r1 -n1 f(" ", 1, " ", 2, " ")')
    _ip.run_line_magic('timeit', '-r1 -n1 ("a " + "b")')
    _ip.run_line_magic('timeit', '-r1 -n1 f("a " + "b")')
    _ip.run_line_magic('timeit', '-r1 -n1 f("a " + "b ")')

def test_timeit_special_syntax():
    if False:
        print('Hello World!')
    'Test %%timeit with IPython special syntax'

    @register_line_magic
    def lmagic(line):
        if False:
            for i in range(10):
                print('nop')
        ip = get_ipython()
        ip.user_ns['lmagic_out'] = line
    _ip.run_line_magic('timeit', '-n1 -r1 %lmagic my line')
    assert _ip.user_ns['lmagic_out'] == 'my line'
    _ip.run_cell_magic('timeit', '-n1 -r1', '%lmagic my line2')
    assert _ip.user_ns['lmagic_out'] == 'my line2'

def test_timeit_return():
    if False:
        i = 10
        return i + 15
    '\n    test whether timeit -o return object\n    '
    res = _ip.run_line_magic('timeit', '-n10 -r10 -o 1')
    assert res is not None

def test_timeit_quiet():
    if False:
        print('Hello World!')
    '\n    test quiet option of timeit magic\n    '
    with tt.AssertNotPrints('loops'):
        _ip.run_cell('%timeit -n1 -r1 -q 1')

def test_timeit_return_quiet():
    if False:
        while True:
            i = 10
    with tt.AssertNotPrints('loops'):
        res = _ip.run_line_magic('timeit', '-n1 -r1 -q -o 1')
    assert res is not None

def test_timeit_invalid_return():
    if False:
        while True:
            i = 10
    with pytest.raises(SyntaxError):
        _ip.run_line_magic('timeit', 'return')

@dec.skipif(execution.profile is None)
def test_prun_special_syntax():
    if False:
        i = 10
        return i + 15
    'Test %%prun with IPython special syntax'

    @register_line_magic
    def lmagic(line):
        if False:
            while True:
                i = 10
        ip = get_ipython()
        ip.user_ns['lmagic_out'] = line
    _ip.run_line_magic('prun', '-q %lmagic my line')
    assert _ip.user_ns['lmagic_out'] == 'my line'
    _ip.run_cell_magic('prun', '-q', '%lmagic my line2')
    assert _ip.user_ns['lmagic_out'] == 'my line2'

@dec.skipif(execution.profile is None)
def test_prun_quotes():
    if False:
        return 10
    'Test that prun does not clobber string escapes (GH #1302)'
    _ip.magic("prun -q x = '\\t'")
    assert _ip.user_ns['x'] == '\t'

def test_extension():
    if False:
        i = 10
        return i + 15
    print('sys.path:')
    for p in sys.path:
        print(' ', p)
    print('CWD', os.getcwd())
    pytest.raises(ImportError, _ip.magic, 'load_ext daft_extension')
    daft_path = os.path.join(os.path.dirname(__file__), 'daft_extension')
    sys.path.insert(0, daft_path)
    try:
        _ip.user_ns.pop('arq', None)
        invalidate_caches()
        _ip.run_line_magic('load_ext', 'daft_extension')
        assert _ip.user_ns['arq'] == 185
        _ip.run_line_magic('unload_ext', 'daft_extension')
        assert 'arq' not in _ip.user_ns
    finally:
        sys.path.remove(daft_path)

def test_notebook_export_json():
    if False:
        return 10
    pytest.importorskip('nbformat')
    _ip = get_ipython()
    _ip.history_manager.reset()
    cmds = ['a=1', 'def b():\n  return a**2', "print('noël, été', b())"]
    for (i, cmd) in enumerate(cmds, start=1):
        _ip.history_manager.store_inputs(i, cmd)
    with TemporaryDirectory() as td:
        outfile = os.path.join(td, 'nb.ipynb')
        _ip.run_line_magic('notebook', '%s' % outfile)

class TestEnv(TestCase):

    def test_env(self):
        if False:
            print('Hello World!')
        env = _ip.run_line_magic('env', '')
        self.assertTrue(isinstance(env, dict))

    def test_env_secret(self):
        if False:
            while True:
                i = 10
        env = _ip.run_line_magic('env', '')
        hidden = '<hidden>'
        with mock.patch.dict(os.environ, {'API_KEY': 'abc123', 'SECRET_THING': 'ssshhh', 'JUPYTER_TOKEN': '', 'VAR': 'abc'}):
            env = _ip.run_line_magic('env', '')
        assert env['API_KEY'] == hidden
        assert env['SECRET_THING'] == hidden
        assert env['JUPYTER_TOKEN'] == hidden
        assert env['VAR'] == 'abc'

    def test_env_get_set_simple(self):
        if False:
            print('Hello World!')
        env = _ip.run_line_magic('env', 'var val1')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val1')
        self.assertEqual(_ip.run_line_magic('env', 'var'), 'val1')
        env = _ip.run_line_magic('env', 'var=val2')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val2')

    def test_env_get_set_complex(self):
        if False:
            while True:
                i = 10
        env = _ip.run_line_magic('env', "var 'val1 '' 'val2")
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], "'val1 '' 'val2")
        self.assertEqual(_ip.run_line_magic('env', 'var'), "'val1 '' 'val2")
        env = _ip.run_line_magic('env', 'var=val2 val3="val4')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val2 val3="val4')

    def test_env_set_bad_input(self):
        if False:
            return 10
        self.assertRaises(UsageError, lambda : _ip.run_line_magic('set_env', 'var'))

    def test_env_set_whitespace(self):
        if False:
            print('Hello World!')
        self.assertRaises(UsageError, lambda : _ip.run_line_magic('env', 'var A=B'))

class CellMagicTestCase(TestCase):

    def check_ident(self, magic):
        if False:
            print('Hello World!')
        out = _ip.run_cell_magic(magic, 'a', 'b')
        assert out == ('a', 'b')
        _ip.run_cell('%%' + magic + ' c\nd\n')
        assert _ip.user_ns['_'] == ('c', 'd\n')

    def test_cell_magic_func_deco(self):
        if False:
            while True:
                i = 10
        'Cell magic using simple decorator'

        @register_cell_magic
        def cellm(line, cell):
            if False:
                for i in range(10):
                    print('nop')
            return (line, cell)
        self.check_ident('cellm')

    def test_cell_magic_reg(self):
        if False:
            for i in range(10):
                print('nop')
        'Cell magic manually registered'

        def cellm(line, cell):
            if False:
                while True:
                    i = 10
            return (line, cell)
        _ip.register_magic_function(cellm, 'cell', 'cellm2')
        self.check_ident('cellm2')

    def test_cell_magic_class(self):
        if False:
            i = 10
            return i + 15
        'Cell magics declared via a class'

        @magics_class
        class MyMagics(Magics):

            @cell_magic
            def cellm3(self, line, cell):
                if False:
                    return 10
                return (line, cell)
        _ip.register_magics(MyMagics)
        self.check_ident('cellm3')

    def test_cell_magic_class2(self):
        if False:
            i = 10
            return i + 15
        'Cell magics declared via a class, #2'

        @magics_class
        class MyMagics2(Magics):

            @cell_magic('cellm4')
            def cellm33(self, line, cell):
                if False:
                    print('Hello World!')
                return (line, cell)
        _ip.register_magics(MyMagics2)
        self.check_ident('cellm4')
        c33 = _ip.find_cell_magic('cellm33')
        assert c33 == None

def test_file():
    if False:
        print('Hello World!')
    'Basic %%writefile'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'file1')
        ip.run_cell_magic('writefile', fname, '\n'.join(['line1', 'line2']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line2' in s

@dec.skip_win32
def test_file_single_quote():
    if False:
        while True:
            i = 10
    'Basic %%writefile with embedded single quotes'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, "'file1'")
        ip.run_cell_magic('writefile', fname, '\n'.join(['line1', 'line2']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line2' in s

@dec.skip_win32
def test_file_double_quote():
    if False:
        for i in range(10):
            print('nop')
    'Basic %%writefile with embedded double quotes'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, '"file1"')
        ip.run_cell_magic('writefile', fname, '\n'.join(['line1', 'line2']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line2' in s

def test_file_var_expand():
    if False:
        while True:
            i = 10
    '%%writefile $filename'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'file1')
        ip.user_ns['filename'] = fname
        ip.run_cell_magic('writefile', '$filename', '\n'.join(['line1', 'line2']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line2' in s

def test_file_unicode():
    if False:
        i = 10
        return i + 15
    '%%writefile with unicode cell'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'file1')
        ip.run_cell_magic('writefile', fname, u'\n'.join([u'liné1', u'liné2']))
        with io.open(fname, encoding='utf-8') as f:
            s = f.read()
        assert 'liné1\n' in s
        assert 'liné2' in s

def test_file_amend():
    if False:
        return 10
    '%%writefile -a amends files'
    ip = get_ipython()
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'file2')
        ip.run_cell_magic('writefile', fname, '\n'.join(['line1', 'line2']))
        ip.run_cell_magic('writefile', '-a %s' % fname, '\n'.join(['line3', 'line4']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line3\n' in s

def test_file_spaces():
    if False:
        print('Hello World!')
    '%%file with spaces in filename'
    ip = get_ipython()
    with TemporaryWorkingDirectory() as td:
        fname = 'file name'
        ip.run_cell_magic('file', '"%s"' % fname, '\n'.join(['line1', 'line2']))
        s = Path(fname).read_text(encoding='utf-8')
        assert 'line1\n' in s
        assert 'line2' in s

def test_script_config():
    if False:
        i = 10
        return i + 15
    ip = get_ipython()
    ip.config.ScriptMagics.script_magics = ['whoda']
    sm = script.ScriptMagics(shell=ip)
    assert 'whoda' in sm.magics['cell']

def test_script_out():
    if False:
        while True:
            i = 10
    ip = get_ipython()
    ip.run_cell_magic('script', f'--out output {sys.executable}', "print('hi')")
    assert ip.user_ns['output'].strip() == 'hi'

def test_script_err():
    if False:
        while True:
            i = 10
    ip = get_ipython()
    ip.run_cell_magic('script', f'--err error {sys.executable}', "import sys; print('hello', file=sys.stderr)")
    assert ip.user_ns['error'].strip() == 'hello'

def test_script_out_err():
    if False:
        i = 10
        return i + 15
    ip = get_ipython()
    ip.run_cell_magic('script', f'--out output --err error {sys.executable}', '\n'.join(['import sys', "print('hi')", "print('hello', file=sys.stderr)"]))
    assert ip.user_ns['output'].strip() == 'hi'
    assert ip.user_ns['error'].strip() == 'hello'

async def test_script_bg_out():
    ip = get_ipython()
    ip.run_cell_magic('script', f'--bg --out output {sys.executable}', "print('hi')")
    assert (await ip.user_ns['output'].read()).strip() == b'hi'
    assert ip.user_ns['output'].at_eof()

async def test_script_bg_err():
    ip = get_ipython()
    ip.run_cell_magic('script', f'--bg --err error {sys.executable}', "import sys; print('hello', file=sys.stderr)")
    assert (await ip.user_ns['error'].read()).strip() == b'hello'
    assert ip.user_ns['error'].at_eof()

async def test_script_bg_out_err():
    ip = get_ipython()
    ip.run_cell_magic('script', f'--bg --out output --err error {sys.executable}', '\n'.join(['import sys', "print('hi')", "print('hello', file=sys.stderr)"]))
    assert (await ip.user_ns['output'].read()).strip() == b'hi'
    assert (await ip.user_ns['error'].read()).strip() == b'hello'
    assert ip.user_ns['output'].at_eof()
    assert ip.user_ns['error'].at_eof()

async def test_script_bg_proc():
    ip = get_ipython()
    ip.run_cell_magic('script', f'--bg --out output --proc p {sys.executable}', '\n'.join(['import sys', "print('hi')", "print('hello', file=sys.stderr)"]))
    p = ip.user_ns['p']
    await p.wait()
    assert p.returncode == 0
    assert (await p.stdout.read()).strip() == b'hi'
    assert await p.stderr.read() == b''
    assert p.stdout.at_eof()
    assert p.stderr.at_eof()

def test_script_defaults():
    if False:
        i = 10
        return i + 15
    ip = get_ipython()
    for cmd in ['sh', 'bash', 'perl', 'ruby']:
        try:
            find_cmd(cmd)
        except Exception:
            pass
        else:
            assert cmd in ip.magics_manager.magics['cell']

@magics_class
class FooFoo(Magics):
    """class with both %foo and %%foo magics"""

    @line_magic('foo')
    def line_foo(self, line):
        if False:
            while True:
                i = 10
        'I am line foo'
        pass

    @cell_magic('foo')
    def cell_foo(self, line, cell):
        if False:
            return 10
        'I am cell foo, not line foo'
        pass

def test_line_cell_info():
    if False:
        for i in range(10):
            print('nop')
    '%%foo and %foo magics are distinguishable to inspect'
    ip = get_ipython()
    ip.magics_manager.register(FooFoo)
    oinfo = ip.object_inspect('foo')
    assert oinfo['found'] is True
    assert oinfo['ismagic'] is True
    oinfo = ip.object_inspect('%%foo')
    assert oinfo['found'] is True
    assert oinfo['ismagic'] is True
    assert oinfo['docstring'] == FooFoo.cell_foo.__doc__
    oinfo = ip.object_inspect('%foo')
    assert oinfo['found'] is True
    assert oinfo['ismagic'] is True
    assert oinfo['docstring'] == FooFoo.line_foo.__doc__

def test_multiple_magics():
    if False:
        for i in range(10):
            print('nop')
    ip = get_ipython()
    foo1 = FooFoo(ip)
    foo2 = FooFoo(ip)
    mm = ip.magics_manager
    mm.register(foo1)
    assert mm.magics['line']['foo'].__self__ is foo1
    mm.register(foo2)
    assert mm.magics['line']['foo'].__self__ is foo2

def test_alias_magic():
    if False:
        print('Hello World!')
    'Test %alias_magic.'
    ip = get_ipython()
    mm = ip.magics_manager
    ip.run_line_magic('alias_magic', 'timeit_alias timeit')
    assert 'timeit_alias' in mm.magics['line']
    assert 'timeit_alias' in mm.magics['cell']
    ip.run_line_magic('alias_magic', '--cell timeit_cell_alias timeit')
    assert 'timeit_cell_alias' not in mm.magics['line']
    assert 'timeit_cell_alias' in mm.magics['cell']
    ip.run_line_magic('alias_magic', '--line env_alias env')
    assert ip.run_line_magic('env', '') == ip.run_line_magic('env_alias', '')
    ip.run_line_magic('alias_magic', '--line history_alias history --params ' + shlex.quote('3'))
    assert 'history_alias' in mm.magics['line']

def test_save():
    if False:
        print('Hello World!')
    'Test %save.'
    ip = get_ipython()
    ip.history_manager.reset()
    cmds = ['a=1', 'def b():\n  return a**2', 'print(a, b())']
    for (i, cmd) in enumerate(cmds, start=1):
        ip.history_manager.store_inputs(i, cmd)
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, 'testsave.py')
        ip.run_line_magic('save', '%s 1-10' % file)
        content = Path(file).read_text(encoding='utf-8')
        assert content.count(cmds[0]) == 1
        assert 'coding: utf-8' in content
        ip.run_line_magic('save', '-a %s 1-10' % file)
        content = Path(file).read_text(encoding='utf-8')
        assert content.count(cmds[0]) == 2
        assert 'coding: utf-8' in content

def test_save_with_no_args():
    if False:
        return 10
    ip = get_ipython()
    ip.history_manager.reset()
    cmds = ['a=1', 'def b():\n    return a**2', 'print(a, b())', '%save']
    for (i, cmd) in enumerate(cmds, start=1):
        ip.history_manager.store_inputs(i, cmd)
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'testsave.py')
        ip.run_line_magic('save', path)
        content = Path(path).read_text(encoding='utf-8')
        expected_content = dedent('            # coding: utf-8\n            a=1\n            def b():\n                return a**2\n            print(a, b())\n            ')
        assert content == expected_content

def test_store():
    if False:
        print('Hello World!')
    'Test %store.'
    ip = get_ipython()
    ip.run_line_magic('load_ext', 'storemagic')
    ip.run_line_magic('store', '-z')
    ip.user_ns['var'] = 42
    ip.run_line_magic('store', 'var')
    ip.user_ns['var'] = 39
    ip.run_line_magic('store', '-r')
    assert ip.user_ns['var'] == 42
    ip.run_line_magic('store', '-d var')
    ip.user_ns['var'] = 39
    ip.run_line_magic('store', '-r')
    assert ip.user_ns['var'] == 39

def _run_edit_test(arg_s, exp_filename=None, exp_lineno=-1, exp_contents=None, exp_is_temp=None):
    if False:
        while True:
            i = 10
    ip = get_ipython()
    M = code.CodeMagics(ip)
    last_call = ['', '']
    (opts, args) = M.parse_options(arg_s, 'prxn:')
    (filename, lineno, is_temp) = M._find_edit_target(ip, args, opts, last_call)
    if exp_filename is not None:
        assert exp_filename == filename
    if exp_contents is not None:
        with io.open(filename, 'r', encoding='utf-8') as f:
            contents = f.read()
        assert exp_contents == contents
    if exp_lineno != -1:
        assert exp_lineno == lineno
    if exp_is_temp is not None:
        assert exp_is_temp == is_temp

def test_edit_interactive():
    if False:
        return 10
    '%edit on interactively defined objects'
    ip = get_ipython()
    n = ip.execution_count
    ip.run_cell('def foo(): return 1', store_history=True)
    with pytest.raises(code.InteractivelyDefined) as e:
        _run_edit_test('foo')
    assert e.value.index == n

def test_edit_cell():
    if False:
        for i in range(10):
            print('nop')
    '%edit [cell id]'
    ip = get_ipython()
    ip.run_cell('def foo(): return 1', store_history=True)
    _run_edit_test('1', exp_contents=ip.user_ns['In'][1], exp_is_temp=True)

def test_edit_fname():
    if False:
        return 10
    '%edit file'
    _run_edit_test('test file.py', exp_filename='test file.py')

def test_bookmark():
    if False:
        print('Hello World!')
    ip = get_ipython()
    ip.run_line_magic('bookmark', 'bmname')
    with tt.AssertPrints('bmname'):
        ip.run_line_magic('bookmark', '-l')
    ip.run_line_magic('bookmark', '-d bmname')

def test_ls_magic():
    if False:
        for i in range(10):
            print('nop')
    ip = get_ipython()
    json_formatter = ip.display_formatter.formatters['application/json']
    json_formatter.enabled = True
    lsmagic = ip.run_line_magic('lsmagic', '')
    with warnings.catch_warnings(record=True) as w:
        j = json_formatter(lsmagic)
    assert sorted(j) == ['cell', 'line']
    assert w == []

def test_strip_initial_indent():
    if False:
        i = 10
        return i + 15

    def sii(s):
        if False:
            for i in range(10):
                print('nop')
        lines = s.splitlines()
        return '\n'.join(code.strip_initial_indent(lines))
    assert sii('  a = 1\nb = 2') == 'a = 1\nb = 2'
    assert sii('  a\n    b\nc') == 'a\n  b\nc'
    assert sii('a\n  b') == 'a\n  b'

def test_logging_magic_quiet_from_arg():
    if False:
        return 10
    _ip.config.LoggingMagics.quiet = False
    lm = logging.LoggingMagics(shell=_ip)
    with TemporaryDirectory() as td:
        try:
            with tt.AssertNotPrints(re.compile('Activating.*')):
                lm.logstart('-q {}'.format(os.path.join(td, 'quiet_from_arg.log')))
        finally:
            _ip.logger.logstop()

def test_logging_magic_quiet_from_config():
    if False:
        return 10
    _ip.config.LoggingMagics.quiet = True
    lm = logging.LoggingMagics(shell=_ip)
    with TemporaryDirectory() as td:
        try:
            with tt.AssertNotPrints(re.compile('Activating.*')):
                lm.logstart(os.path.join(td, 'quiet_from_config.log'))
        finally:
            _ip.logger.logstop()

def test_logging_magic_not_quiet():
    if False:
        for i in range(10):
            print('nop')
    _ip.config.LoggingMagics.quiet = False
    lm = logging.LoggingMagics(shell=_ip)
    with TemporaryDirectory() as td:
        try:
            with tt.AssertPrints(re.compile('Activating.*')):
                lm.logstart(os.path.join(td, 'not_quiet.log'))
        finally:
            _ip.logger.logstop()

def test_time_no_var_expand():
    if False:
        while True:
            i = 10
    _ip.user_ns['a'] = 5
    _ip.user_ns['b'] = []
    _ip.run_line_magic('time', 'b.append("{a}")')
    assert _ip.user_ns['b'] == ['{a}']

def test_timeit_arguments():
    if False:
        return 10
    'Test valid timeit arguments, should not cause SyntaxError (GH #1269)'
    _ip.run_line_magic('timeit', "-n1 -r1 a=('#')")
MINIMAL_LAZY_MAGIC = '\nfrom IPython.core.magic import (\n    Magics,\n    magics_class,\n    line_magic,\n    cell_magic,\n)\n\n\n@magics_class\nclass LazyMagics(Magics):\n    @line_magic\n    def lazy_line(self, line):\n        print("Lazy Line")\n\n    @cell_magic\n    def lazy_cell(self, line, cell):\n        print("Lazy Cell")\n\n\ndef load_ipython_extension(ipython):\n    ipython.register_magics(LazyMagics)\n'

def test_lazy_magics():
    if False:
        print('Hello World!')
    with pytest.raises(UsageError):
        ip.run_line_magic('lazy_line', '')
    startdir = os.getcwd()
    with TemporaryDirectory() as tmpdir:
        with prepended_to_syspath(tmpdir):
            ptempdir = Path(tmpdir)
            tf = ptempdir / 'lazy_magic_module.py'
            tf.write_text(MINIMAL_LAZY_MAGIC)
            ip.magics_manager.register_lazy('lazy_line', Path(tf.name).name[:-3])
            with tt.AssertPrints('Lazy Line'):
                ip.run_line_magic('lazy_line', '')
TEST_MODULE = '\nprint(\'Loaded my_tmp\')\nif __name__ == "__main__":\n    print(\'I just ran a script\')\n'

def test_run_module_from_import_hook():
    if False:
        for i in range(10):
            print('nop')
    'Test that a module can be loaded via an import hook'
    with TemporaryDirectory() as tmpdir:
        fullpath = os.path.join(tmpdir, 'my_tmp.py')
        Path(fullpath).write_text(TEST_MODULE, encoding='utf-8')
        import importlib.abc
        import importlib.util

        class MyTempImporter(importlib.abc.MetaPathFinder, importlib.abc.SourceLoader):

            def find_spec(self, fullname, path, target=None):
                if False:
                    for i in range(10):
                        print('nop')
                if fullname == 'my_tmp':
                    return importlib.util.spec_from_loader(fullname, self)

            def get_filename(self, fullname):
                if False:
                    return 10
                assert fullname == 'my_tmp'
                return fullpath

            def get_data(self, path):
                if False:
                    i = 10
                    return i + 15
                assert Path(path).samefile(fullpath)
                return Path(fullpath).read_text(encoding='utf-8')
        sys.meta_path.insert(0, MyTempImporter())
        with capture_output() as captured:
            _ip.run_line_magic('run', '-m my_tmp')
            _ip.run_cell('import my_tmp')
        output = 'Loaded my_tmp\nI just ran a script\nLoaded my_tmp\n'
        assert output == captured.stdout
        sys.meta_path.pop(0)