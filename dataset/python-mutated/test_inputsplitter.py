"""Tests for the inputsplitter module."""
import unittest
import pytest
import sys
with pytest.warns(DeprecationWarning, match='inputsplitter'):
    from IPython.core import inputsplitter as isp
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt

def mini_interactive_loop(input_func):
    if False:
        print('Hello World!')
    'Minimal example of the logic of an interactive interpreter loop.\n\n    This serves as an example, and it is used by the test system with a fake\n    raw_input that simulates interactive input.'
    from IPython.core.inputsplitter import InputSplitter
    isp = InputSplitter()
    while isp.push_accepts_more():
        indent = ' ' * isp.get_indent_spaces()
        prompt = '>>> ' + indent
        line = indent + input_func(prompt)
        isp.push(line)
    src = isp.source_reset()
    return src

def pseudo_input(lines):
    if False:
        for i in range(10):
            print('nop')
    'Return a function that acts like raw_input but feeds the input list.'
    ilines = iter(lines)

    def raw_in(prompt):
        if False:
            i = 10
            return i + 15
        try:
            return next(ilines)
        except StopIteration:
            return ''
    return raw_in

def test_spaces():
    if False:
        print('Hello World!')
    tests = [('', 0), (' ', 1), ('\n', 0), (' \n', 1), ('x', 0), (' x', 1), ('  x', 2), ('    x', 4), ('\tx', 1), ('\t x', 2)]
    with pytest.warns(PendingDeprecationWarning):
        tt.check_pairs(isp.num_ini_spaces, tests)

def test_remove_comments():
    if False:
        for i in range(10):
            print('nop')
    tests = [('text', 'text'), ('text # comment', 'text '), ('text # comment\n', 'text \n'), ('text # comment \n', 'text \n'), ('line # c \nline\n', 'line \nline\n'), ('line # c \nline#c2  \nline\nline #c\n\n', 'line \nline\nline\nline \n\n')]
    tt.check_pairs(isp.remove_comments, tests)

def test_get_input_encoding():
    if False:
        print('Hello World!')
    encoding = isp.get_input_encoding()
    assert isinstance(encoding, str)
    assert 'test'.encode(encoding) == b'test'

class NoInputEncodingTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.old_stdin = sys.stdin

        class X:
            pass
        fake_stdin = X()
        sys.stdin = fake_stdin

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        enc = isp.get_input_encoding()
        self.assertEqual(enc, 'ascii')

    def tearDown(self):
        if False:
            print('Hello World!')
        sys.stdin = self.old_stdin

class InputSplitterTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.isp = isp.InputSplitter()

    def test_reset(self):
        if False:
            return 10
        isp = self.isp
        isp.push('x=1')
        isp.reset()
        self.assertEqual(isp._buffer, [])
        self.assertEqual(isp.get_indent_spaces(), 0)
        self.assertEqual(isp.source, '')
        self.assertEqual(isp.code, None)
        self.assertEqual(isp._is_complete, False)

    def test_source(self):
        if False:
            print('Hello World!')
        self.isp._store('1')
        self.isp._store('2')
        self.assertEqual(self.isp.source, '1\n2\n')
        self.assertEqual(len(self.isp._buffer) > 0, True)
        self.assertEqual(self.isp.source_reset(), '1\n2\n')
        self.assertEqual(self.isp._buffer, [])
        self.assertEqual(self.isp.source, '')

    def test_indent(self):
        if False:
            while True:
                i = 10
        isp = self.isp
        isp.push('x=1')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_indent2(self):
        if False:
            return 10
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push(' ' * 2)
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent3(self):
        if False:
            return 10
        isp = self.isp
        isp.push('if 1:')
        isp.push('    x = (1+\n    2)')
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent4(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        isp.push('if 1: \n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\t\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_pass(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        isp.push('if 1:\n    passes = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     pass')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     pass   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_break(self):
        if False:
            for i in range(10):
                print('nop')
        isp = self.isp
        isp.push('while 1:\n    breaks = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     break')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     break   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_continue(self):
        if False:
            while True:
                i = 10
        isp = self.isp
        isp.push('while 1:\n    continues = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     continue')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     continue   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_raise(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        isp.push('if 1:\n    raised = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     raise TypeError()')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise      ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_return(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        isp.push('if 1:\n    returning = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     return 5 + 493')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return      ')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return(0)')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_push(self):
        if False:
            print('Hello World!')
        isp = self.isp
        self.assertEqual(isp.push('x=1'), True)

    def test_push2(self):
        if False:
            return 10
        isp = self.isp
        self.assertEqual(isp.push('if 1:'), False)
        for line in ['  x=1', '# a comment', '  y=2']:
            print(line)
            self.assertEqual(isp.push(line), True)

    def test_push3(self):
        if False:
            while True:
                i = 10
        isp = self.isp
        isp.push('if True:')
        isp.push('  a = 1')
        self.assertEqual(isp.push('b = [1,'), False)

    def test_push_accepts_more(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        isp.push('x=1')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more2(self):
        if False:
            return 10
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('  x=1')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more3(self):
        if False:
            print('Hello World!')
        isp = self.isp
        isp.push('x = (2+\n3)')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more4(self):
        if False:
            return 10
        isp = self.isp
        isp.push('if 1:')
        isp.push('    x = (2+')
        isp.push('    3)')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('    y = 3')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more5(self):
        if False:
            print('Hello World!')
        isp = self.isp
        isp.push('try:')
        isp.push('    a = 5')
        isp.push('except:')
        isp.push('    raise')
        self.assertEqual(isp.push_accepts_more(), True)

    def test_continuation(self):
        if False:
            for i in range(10):
                print('nop')
        isp = self.isp
        isp.push('import os, \\')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('sys')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_syntax_error(self):
        if False:
            return 10
        isp = self.isp
        isp.push('run foo')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        self.isp.push(u'Pérez')
        self.isp.push(u'Ã©')
        self.isp.push(u"u'Ã©'")

    @pytest.mark.xfail(reason='Bug in python 3.9.8 –\xa0bpo 45738', condition=sys.version_info in [(3, 9, 8, 'final', 0), (3, 11, 0, 'alpha', 2)], raises=SystemError, strict=True)
    def test_line_continuation(self):
        if False:
            i = 10
            return i + 15
        ' Test issue #2108.'
        isp = self.isp
        isp.push('1 \\\n\n')
        self.assertEqual(isp.push_accepts_more(), False)
        isp.push('1 \\ ')
        self.assertEqual(isp.push_accepts_more(), False)
        isp.push('(1 \\ ')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_check_complete(self):
        if False:
            for i in range(10):
                print('nop')
        isp = self.isp
        self.assertEqual(isp.check_complete('a = 1'), ('complete', None))
        self.assertEqual(isp.check_complete('for a in range(5):'), ('incomplete', 4))
        self.assertEqual(isp.check_complete('raise = 2'), ('invalid', None))
        self.assertEqual(isp.check_complete('a = [1,\n2,'), ('incomplete', 0))
        self.assertEqual(isp.check_complete('def a():\n x=1\n global x'), ('invalid', None))

class InteractiveLoopTestCase(unittest.TestCase):
    """Tests for an interactive loop like a python shell.
    """

    def check_ns(self, lines, ns):
        if False:
            for i in range(10):
                print('nop')
        'Validate that the given input lines produce the resulting namespace.\n\n        Note: the input lines are given exactly as they would be typed in an\n        auto-indenting environment, as mini_interactive_loop above already does\n        auto-indenting and prepends spaces to the input.\n        '
        src = mini_interactive_loop(pseudo_input(lines))
        test_ns = {}
        exec(src, test_ns)
        for (k, v) in ns.items():
            self.assertEqual(test_ns[k], v)

    def test_simple(self):
        if False:
            while True:
                i = 10
        self.check_ns(['x=1'], dict(x=1))

    def test_simple2(self):
        if False:
            i = 10
            return i + 15
        self.check_ns(['if 1:', 'x=2'], dict(x=2))

    def test_xy(self):
        if False:
            print('Hello World!')
        self.check_ns(['x=1; y=2'], dict(x=1, y=2))

    def test_abc(self):
        if False:
            i = 10
            return i + 15
        self.check_ns(['if 1:', 'a=1', 'b=2', 'c=3'], dict(a=1, b=2, c=3))

    def test_multi(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_ns(['x =(1+', '1+', '2)'], dict(x=4))

class IPythonInputTestCase(InputSplitterTestCase):
    """By just creating a new class whose .isp is a different instance, we
    re-run the same test battery on the new input splitter.

    In addition, this runs the tests over the syntax and syntax_ml dicts that
    were tested by individual functions, as part of the OO interface.

    It also makes some checks on the raw buffer storage.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.isp = isp.IPythonInputSplitter()

    def test_syntax(self):
        if False:
            return 10
        'Call all single-line syntax tests from the main object'
        isp = self.isp
        for example in syntax.values():
            for (raw, out_t) in example:
                if raw.startswith(' '):
                    continue
                isp.push(raw + '\n')
                out_raw = isp.source_raw
                out = isp.source_reset()
                self.assertEqual(out.rstrip(), out_t, tt.pair_fail_msg.format('inputsplitter', raw, out_t, out))
                self.assertEqual(out_raw.rstrip(), raw.rstrip())

    def test_syntax_multiline(self):
        if False:
            print('Hello World!')
        isp = self.isp
        for example in syntax_ml.values():
            for line_pairs in example:
                out_t_parts = []
                raw_parts = []
                for (lraw, out_t_part) in line_pairs:
                    if out_t_part is not None:
                        out_t_parts.append(out_t_part)
                    if lraw is not None:
                        isp.push(lraw)
                        raw_parts.append(lraw)
                out_raw = isp.source_raw
                out = isp.source_reset()
                out_t = '\n'.join(out_t_parts).rstrip()
                raw = '\n'.join(raw_parts).rstrip()
                self.assertEqual(out.rstrip(), out_t)
                self.assertEqual(out_raw.rstrip(), raw)

    def test_syntax_multiline_cell(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp
        for example in syntax_ml.values():
            out_t_parts = []
            for line_pairs in example:
                raw = '\n'.join((r for (r, _) in line_pairs if r is not None))
                out_t = '\n'.join((t for (_, t) in line_pairs if t is not None))
                out = isp.transform_cell(raw)
                self.assertEqual(out.rstrip(), out_t.rstrip())

    def test_cellmagic_preempt(self):
        if False:
            return 10
        isp = self.isp
        for (raw, name, line, cell) in [('%%cellm a\nIn[1]:', u'cellm', u'a', u'In[1]:'), ('%%cellm \nline\n>>> hi', u'cellm', u'', u'line\n>>> hi'), ('>>> %%cellm \nline\n>>> hi', u'cellm', u'', u'line\nhi'), ('%%cellm \n>>> hi', u'cellm', u'', u'>>> hi'), ('%%cellm \nline1\nline2', u'cellm', u'', u'line1\nline2'), ('%%cellm \nline1\\\\\nline2', u'cellm', u'', u'line1\\\\\nline2')]:
            expected = 'get_ipython().run_cell_magic(%r, %r, %r)' % (name, line, cell)
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())

    def test_multiline_passthrough(self):
        if False:
            i = 10
            return i + 15
        isp = self.isp

        class CommentTransformer(InputTransformer):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self._lines = []

            def push(self, line):
                if False:
                    i = 10
                    return i + 15
                self._lines.append(line + '#')

            def reset(self):
                if False:
                    while True:
                        i = 10
                text = '\n'.join(self._lines)
                self._lines = []
                return text
        isp.physical_line_transforms.insert(0, CommentTransformer())
        for (raw, expected) in [('a=5', 'a=5#'), ('%ls foo', 'get_ipython().run_line_magic(%r, %r)' % (u'ls', u'foo#')), ('!ls foo\n%ls bar', 'get_ipython().system(%r)\nget_ipython().run_line_magic(%r, %r)' % (u'ls foo#', u'ls', u'bar#')), ('1\n2\n3\n%ls foo\n4\n5', '1#\n2#\n3#\nget_ipython().run_line_magic(%r, %r)\n4#\n5#' % (u'ls', u'foo#'))]:
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())
if __name__ == '__main__':
    from IPython.core.inputsplitter import IPythonInputSplitter
    (isp, start_prompt) = (IPythonInputSplitter(), 'In> ')
    autoindent = True
    try:
        while True:
            prompt = start_prompt
            while isp.push_accepts_more():
                indent = ' ' * isp.get_indent_spaces()
                if autoindent:
                    line = indent + input(prompt + indent)
                else:
                    line = input(prompt)
                isp.push(line)
                prompt = '... '
            raw = isp.source_raw
            src = isp.source_reset()
            print('Input source was:\n', src)
            print('Raw source was:\n', raw)
    except EOFError:
        print('Bye')

def test_last_blank():
    if False:
        for i in range(10):
            print('nop')
    assert isp.last_blank('') is False
    assert isp.last_blank('abc') is False
    assert isp.last_blank('abc\n') is False
    assert isp.last_blank('abc\na') is False
    assert isp.last_blank('\n') is True
    assert isp.last_blank('\n ') is True
    assert isp.last_blank('abc\n ') is True
    assert isp.last_blank('abc\n\n') is True
    assert isp.last_blank('abc\nd\n\n') is True
    assert isp.last_blank('abc\nd\ne\n\n') is True
    assert isp.last_blank('abc \n \n \n\n') is True

def test_last_two_blanks():
    if False:
        while True:
            i = 10
    assert isp.last_two_blanks('') is False
    assert isp.last_two_blanks('abc') is False
    assert isp.last_two_blanks('abc\n') is False
    assert isp.last_two_blanks('abc\n\na') is False
    assert isp.last_two_blanks('abc\n \n') is False
    assert isp.last_two_blanks('abc\n\n') is False
    assert isp.last_two_blanks('\n\n') is True
    assert isp.last_two_blanks('\n\n ') is True
    assert isp.last_two_blanks('\n \n') is True
    assert isp.last_two_blanks('abc\n\n ') is True
    assert isp.last_two_blanks('abc\n\n\n') is True
    assert isp.last_two_blanks('abc\n\n \n') is True
    assert isp.last_two_blanks('abc\n\n \n ') is True
    assert isp.last_two_blanks('abc\n\n \n \n') is True
    assert isp.last_two_blanks('abc\nd\n\n\n') is True
    assert isp.last_two_blanks('abc\nd\ne\nf\n\n\n') is True

class CellMagicsCommon(object):

    def test_whole_cell(self):
        if False:
            for i in range(10):
                print('nop')
        src = '%%cellm line\nbody\n'
        out = self.sp.transform_cell(src)
        ref = "get_ipython().run_cell_magic('cellm', 'line', 'body')\n"
        assert out == ref

    def test_cellmagic_help(self):
        if False:
            return 10
        self.sp.push('%%cellm?')
        assert self.sp.push_accepts_more() is False

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.sp.reset()

class CellModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=False)

    def test_incremental(self):
        if False:
            while True:
                i = 10
        sp = self.sp
        sp.push('%%cellm firstline\n')
        assert sp.push_accepts_more() is True
        sp.push('line2\n')
        assert sp.push_accepts_more() is True
        sp.push('\n')
        assert sp.push_accepts_more() is True

    def test_no_strip_coding(self):
        if False:
            return 10
        src = '\n'.join(['%%writefile foo.py', '# coding: utf-8', 'print(u"üñîçø∂é")'])
        out = self.sp.transform_cell(src)
        assert '# coding: utf-8' in out

class LineModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=True)

    def test_incremental(self):
        if False:
            i = 10
            return i + 15
        sp = self.sp
        sp.push('%%cellm line2\n')
        assert sp.push_accepts_more() is True
        sp.push('\n')
        assert sp.push_accepts_more() is False
indentation_samples = [('a = 1', 0), ('for a in b:', 4), ('def f():', 4), ('def f(): #comment', 4), ('a = ":#not a comment"', 0), ('def f():\n    a = 1', 4), ('def f():\n    return 1', 0), ('for a in b:\n   if a < 0:       continue', 3), ('a = {', 4), ('a = {\n     1,', 5), ('b = """123', 0), ('', 0), ('def f():\n    pass', 0), ('class Bar:\n    def f():\n        pass', 4), ('class Bar:\n    def f():\n        raise', 4)]

def test_find_next_indent():
    if False:
        print('Hello World!')
    for (code, exp) in indentation_samples:
        res = isp.find_next_indent(code)
        msg = '{!r} != {!r} (expected)\n Code: {!r}'.format(res, exp, code)
        assert res == exp, msg