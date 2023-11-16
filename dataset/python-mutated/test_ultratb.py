"""Tests for IPython.core.ultratb
"""
import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths
from IPython.utils.syspathcontext import prepended_to_syspath
file_1 = '1\n2\n3\ndef f():\n  1/0\n'
file_2 = 'def f():\n  1/0\n'

def recursionlimit(frames):
    if False:
        while True:
            i = 10
    '\n    decorator to set the recursion limit temporarily\n    '

    def inner(test_function):
        if False:
            for i in range(10):
                print('nop')

        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            rl = sys.getrecursionlimit()
            sys.setrecursionlimit(frames)
            try:
                return test_function(*args, **kwargs)
            finally:
                sys.setrecursionlimit(rl)
        return wrapper
    return inner

class ChangedPyFileTest(unittest.TestCase):

    def test_changing_py_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Traceback produced if the line where the error occurred is missing?\n\n        https://github.com/ipython/ipython/issues/1456\n        '
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'foo.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(file_1)
            with prepended_to_syspath(td):
                ip.run_cell('import foo')
            with tt.AssertPrints('ZeroDivisionError'):
                ip.run_cell('foo.f()')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(file_2)
            with tt.AssertNotPrints('Internal Python error', channel='stderr'):
                with tt.AssertPrints('ZeroDivisionError'):
                    ip.run_cell('foo.f()')
                with tt.AssertPrints('ZeroDivisionError'):
                    ip.run_cell('foo.f()')
iso_8859_5_file = u'# coding: iso-8859-5\n\ndef fail():\n    """дбИЖ"""\n    1/0     # дбИЖ\n'

class NonAsciiTest(unittest.TestCase):

    @onlyif_unicode_paths
    def test_nonascii_path(self):
        if False:
            return 10
        with TemporaryDirectory(suffix=u'é') as td:
            fname = os.path.join(td, u'fooé.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(file_1)
            with prepended_to_syspath(td):
                ip.run_cell('import foo')
            with tt.AssertPrints('ZeroDivisionError'):
                ip.run_cell('foo.f()')

    def test_iso8859_5(self):
        if False:
            return 10
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'dfghjkl.py')
            with io.open(fname, 'w', encoding='iso-8859-5') as f:
                f.write(iso_8859_5_file)
            with prepended_to_syspath(td):
                ip.run_cell('from dfghjkl import fail')
            with tt.AssertPrints('ZeroDivisionError'):
                with tt.AssertPrints(u'дбИЖ', suppress=False):
                    ip.run_cell('fail()')

    def test_nonascii_msg(self):
        if False:
            while True:
                i = 10
        cell = u"raise Exception('é')"
        expected = u"Exception('é')"
        ip.run_cell('%xmode plain')
        with tt.AssertPrints(expected):
            ip.run_cell(cell)
        ip.run_cell('%xmode verbose')
        with tt.AssertPrints(expected):
            ip.run_cell(cell)
        ip.run_cell('%xmode context')
        with tt.AssertPrints(expected):
            ip.run_cell(cell)
        ip.run_cell('%xmode minimal')
        with tt.AssertPrints(u'Exception: é'):
            ip.run_cell(cell)
        ip.run_cell('%xmode context')

class NestedGenExprTestCase(unittest.TestCase):
    """
    Regression test for the following issues:
    https://github.com/ipython/ipython/issues/8293
    https://github.com/ipython/ipython/issues/8205
    """

    def test_nested_genexpr(self):
        if False:
            for i in range(10):
                print('nop')
        code = dedent('            class SpecificException(Exception):\n                pass\n\n            def foo(x):\n                raise SpecificException("Success!")\n\n            sum(sum(foo(x) for _ in [0]) for x in [0])\n            ')
        with tt.AssertPrints('SpecificException: Success!', suppress=False):
            ip.run_cell(code)
indentationerror_file = 'if True:\nzoon()\n'

class IndentationErrorTest(unittest.TestCase):

    def test_indentationerror_shows_line(self):
        if False:
            return 10
        with tt.AssertPrints('IndentationError'):
            with tt.AssertPrints('zoon()', suppress=False):
                ip.run_cell(indentationerror_file)
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'foo.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(indentationerror_file)
            with tt.AssertPrints('IndentationError'):
                with tt.AssertPrints('zoon()', suppress=False):
                    ip.magic('run %s' % fname)
se_file_1 = '1\n2\n7/\n'
se_file_2 = '7/\n'

class SyntaxErrorTest(unittest.TestCase):

    def test_syntaxerror_no_stacktrace_at_compile_time(self):
        if False:
            return 10
        syntax_error_at_compile_time = '\ndef foo():\n    ..\n'
        with tt.AssertPrints('SyntaxError'):
            ip.run_cell(syntax_error_at_compile_time)
        with tt.AssertNotPrints('foo()'):
            ip.run_cell(syntax_error_at_compile_time)

    def test_syntaxerror_stacktrace_when_running_compiled_code(self):
        if False:
            while True:
                i = 10
        syntax_error_at_runtime = '\ndef foo():\n    eval("..")\n\ndef bar():\n    foo()\n\nbar()\n'
        with tt.AssertPrints('SyntaxError'):
            ip.run_cell(syntax_error_at_runtime)
        with tt.AssertPrints(['foo()', 'bar()']):
            ip.run_cell(syntax_error_at_runtime)
        del ip.user_ns['bar']
        del ip.user_ns['foo']

    def test_changing_py_file(self):
        if False:
            for i in range(10):
                print('nop')
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'foo.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(se_file_1)
            with tt.AssertPrints(['7/', 'SyntaxError']):
                ip.magic('run ' + fname)
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(se_file_2)
            with tt.AssertPrints(['7/', 'SyntaxError']):
                ip.magic('run ' + fname)

    def test_non_syntaxerror(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise ValueError('QWERTY')
        except ValueError:
            with tt.AssertPrints('QWERTY'):
                ip.showsyntaxerror()
import sys
if platform.python_implementation() != 'PyPy':
    '\n    New 3.9 Pgen Parser does not raise Memory error, except on failed malloc.\n    '

    class MemoryErrorTest(unittest.TestCase):

        def test_memoryerror(self):
            if False:
                return 10
            memoryerror_code = '(' * 200 + ')' * 200
            ip.run_cell(memoryerror_code)

class Python3ChainedExceptionsTest(unittest.TestCase):
    DIRECT_CAUSE_ERROR_CODE = "\ntry:\n    x = 1 + 2\n    print(not_defined_here)\nexcept Exception as e:\n    x += 55\n    x - 1\n    y = {}\n    raise KeyError('uh') from e\n    "
    EXCEPTION_DURING_HANDLING_CODE = "\ntry:\n    x = 1 + 2\n    print(not_defined_here)\nexcept Exception as e:\n    x += 55\n    x - 1\n    y = {}\n    raise KeyError('uh')\n    "
    SUPPRESS_CHAINING_CODE = '\ntry:\n    1/0\nexcept Exception:\n    raise ValueError("Yikes") from None\n    '

    def test_direct_cause_error(self):
        if False:
            i = 10
            return i + 15
        with tt.AssertPrints(['KeyError', 'NameError', 'direct cause']):
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)

    def test_exception_during_handling_error(self):
        if False:
            while True:
                i = 10
        with tt.AssertPrints(['KeyError', 'NameError', 'During handling']):
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)

    def test_suppress_exception_chaining(self):
        if False:
            for i in range(10):
                print('nop')
        with tt.AssertNotPrints('ZeroDivisionError'), tt.AssertPrints('ValueError', suppress=False):
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)

    def test_plain_direct_cause_error(self):
        if False:
            return 10
        with tt.AssertPrints(['KeyError', 'NameError', 'direct cause']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)
            ip.run_cell('%xmode Verbose')

    def test_plain_exception_during_handling_error(self):
        if False:
            while True:
                i = 10
        with tt.AssertPrints(['KeyError', 'NameError', 'During handling']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)
            ip.run_cell('%xmode Verbose')

    def test_plain_suppress_exception_chaining(self):
        if False:
            return 10
        with tt.AssertNotPrints('ZeroDivisionError'), tt.AssertPrints('ValueError', suppress=False):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)
            ip.run_cell('%xmode Verbose')

class RecursionTest(unittest.TestCase):
    DEFINITIONS = '\ndef non_recurs():\n    1/0\n\ndef r1():\n    r1()\n\ndef r3a():\n    r3b()\n\ndef r3b():\n    r3c()\n\ndef r3c():\n    r3a()\n\ndef r3o1():\n    r3a()\n\ndef r3o2():\n    r3o1()\n'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ip.run_cell(self.DEFINITIONS)

    def test_no_recursion(self):
        if False:
            while True:
                i = 10
        with tt.AssertNotPrints('skipping similar frames'):
            ip.run_cell('non_recurs()')

    @recursionlimit(200)
    def test_recursion_one_frame(self):
        if False:
            print('Hello World!')
        with tt.AssertPrints(re.compile('\\[\\.\\.\\. skipping similar frames: r1 at line 5 \\(\\d{2,3} times\\)\\]')):
            ip.run_cell('r1()')

    @recursionlimit(160)
    def test_recursion_three_frames(self):
        if False:
            while True:
                i = 10
        with tt.AssertPrints('[... skipping similar frames: '), tt.AssertPrints(re.compile('r3a at line 8 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3b at line 11 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3c at line 14 \\(\\d{2} times\\)'), suppress=False):
            ip.run_cell('r3o2()')

class PEP678NotesReportingTest(unittest.TestCase):
    ERROR_WITH_NOTE = '\ntry:\n    raise AssertionError("Message")\nexcept Exception as e:\n    try:\n        e.add_note("This is a PEP-678 note.")\n    except AttributeError:  # Python <= 3.10\n        e.__notes__ = ("This is a PEP-678 note.",)\n    raise\n    '

    def test_verbose_reports_notes(self):
        if False:
            print('Hello World!')
        with tt.AssertPrints(['AssertionError', 'Message', 'This is a PEP-678 note.']):
            ip.run_cell(self.ERROR_WITH_NOTE)

    def test_plain_reports_notes(self):
        if False:
            for i in range(10):
                print('nop')
        with tt.AssertPrints(['AssertionError', 'Message', 'This is a PEP-678 note.']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.ERROR_WITH_NOTE)
            ip.run_cell('%xmode Verbose')

def test_handlers():
    if False:
        print('Hello World!')

    def spam(c, d_e):
        if False:
            for i in range(10):
                print('nop')
        (d, e) = d_e
        x = c + d
        y = c * d
        foo(x, y)

    def foo(a, b, bar=1):
        if False:
            print('Hello World!')
        eggs(a, b + bar)

    def eggs(f, g, z=globals()):
        if False:
            return 10
        h = f + g
        i = f - g
        return h / i
    buff = io.StringIO()
    buff.write('')
    buff.write('*** Before ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        traceback.print_exc(file=buff)
    handler = ColorTB(ostream=buff)
    buff.write('*** ColorTB ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        handler(*sys.exc_info())
    buff.write('')
    handler = VerboseTB(ostream=buff)
    buff.write('*** VerboseTB ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        handler(*sys.exc_info())
    buff.write('')