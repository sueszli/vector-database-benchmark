"""Tests for the key interactiveshell module.

Historically the main classes in interactiveshell have been under-tested.  This
module should grow as many single-method tests as possible to trap many of the
recurring bugs we seem to encounter with high-level interaction.
"""
import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import skipif, skip_win32, onlyif_unicode_paths, onlyif_cmds_exist
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd

class DerivedInterrupt(KeyboardInterrupt):
    pass

class InteractiveShellTestCase(unittest.TestCase):

    def test_naked_string_cells(self):
        if False:
            while True:
                i = 10
        'Test that cells with only naked strings are fully executed'
        ip.run_cell('"a"\n')
        self.assertEqual(ip.user_ns['_'], 'a')
        ip.run_cell('"""a\nb"""\n')
        self.assertEqual(ip.user_ns['_'], 'a\nb')

    def test_run_empty_cell(self):
        if False:
            while True:
                i = 10
        "Just make sure we don't get a horrible error with a blank\n        cell of input. Yes, I did overlook that."
        old_xc = ip.execution_count
        res = ip.run_cell('')
        self.assertEqual(ip.execution_count, old_xc)
        self.assertEqual(res.execution_count, None)

    def test_run_cell_multiline(self):
        if False:
            while True:
                i = 10
        'Multi-block, multi-line cells must execute correctly.\n        '
        src = '\n'.join(['x=1', 'y=2', 'if 1:', '    x += 1', '    y += 1'])
        res = ip.run_cell(src)
        self.assertEqual(ip.user_ns['x'], 2)
        self.assertEqual(ip.user_ns['y'], 3)
        self.assertEqual(res.success, True)
        self.assertEqual(res.result, None)

    def test_multiline_string_cells(self):
        if False:
            return 10
        'Code sprinkled with multiline strings should execute (GH-306)'
        ip.run_cell('tmp=0')
        self.assertEqual(ip.user_ns['tmp'], 0)
        res = ip.run_cell('tmp=1;"""a\nb"""\n')
        self.assertEqual(ip.user_ns['tmp'], 1)
        self.assertEqual(res.success, True)
        self.assertEqual(res.result, 'a\nb')

    def test_dont_cache_with_semicolon(self):
        if False:
            for i in range(10):
                print('nop')
        'Ending a line with semicolon should not cache the returned object (GH-307)'
        oldlen = len(ip.user_ns['Out'])
        for cell in ['1;', '1;1;']:
            res = ip.run_cell(cell, store_history=True)
            newlen = len(ip.user_ns['Out'])
            self.assertEqual(oldlen, newlen)
            self.assertIsNone(res.result)
        i = 0
        for cell in ['1', '1;1']:
            ip.run_cell(cell, store_history=True)
            newlen = len(ip.user_ns['Out'])
            i += 1
            self.assertEqual(oldlen + i, newlen)

    def test_syntax_error(self):
        if False:
            i = 10
            return i + 15
        res = ip.run_cell('raise = 3')
        self.assertIsInstance(res.error_before_exec, SyntaxError)

    def test_open_standard_input_stream(self):
        if False:
            while True:
                i = 10
        res = ip.run_cell('open(0)')
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_open_standard_output_stream(self):
        if False:
            for i in range(10):
                print('nop')
        res = ip.run_cell('open(1)')
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_open_standard_error_stream(self):
        if False:
            i = 10
            return i + 15
        res = ip.run_cell('open(2)')
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_In_variable(self):
        if False:
            while True:
                i = 10
        'Verify that In variable grows with user input (GH-284)'
        oldlen = len(ip.user_ns['In'])
        ip.run_cell('1;', store_history=True)
        newlen = len(ip.user_ns['In'])
        self.assertEqual(oldlen + 1, newlen)
        self.assertEqual(ip.user_ns['In'][-1], '1;')

    def test_magic_names_in_string(self):
        if False:
            return 10
        ip.run_cell('a = """\n%exit\n"""')
        self.assertEqual(ip.user_ns['a'], '\n%exit\n')

    def test_trailing_newline(self):
        if False:
            return 10
        'test that running !(command) does not raise a SyntaxError'
        ip.run_cell('!(true)\n', False)
        ip.run_cell('!(true)\n\n\n', False)

    def test_gh_597(self):
        if False:
            print('Hello World!')
        'Pretty-printing lists of objects with non-ascii reprs may cause\n        problems.'

        class Spam(object):

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return 'é' * 50
        import IPython.core.formatters
        f = IPython.core.formatters.PlainTextFormatter()
        f([Spam(), Spam()])

    def test_future_flags(self):
        if False:
            i = 10
            return i + 15
        'Check that future flags are used for parsing code (gh-777)'
        ip.run_cell('from __future__ import barry_as_FLUFL')
        try:
            ip.run_cell('prfunc_return_val = 1 <> 2')
            assert 'prfunc_return_val' in ip.user_ns
        finally:
            ip.compile.reset_compiler_flags()

    def test_can_pickle(self):
        if False:
            i = 10
            return i + 15
        'Can we pickle objects defined interactively (GH-29)'
        ip = get_ipython()
        ip.reset()
        ip.run_cell('class Mylist(list):\n    def __init__(self,x=[]):\n        list.__init__(self,x)')
        ip.run_cell('w=Mylist([1,2,3])')
        from pickle import dumps
        _main = sys.modules['__main__']
        sys.modules['__main__'] = ip.user_module
        try:
            res = dumps(ip.user_ns['w'])
        finally:
            sys.modules['__main__'] = _main
        self.assertTrue(isinstance(res, bytes))

    def test_global_ns(self):
        if False:
            for i in range(10):
                print('nop')
        'Code in functions must be able to access variables outside them.'
        ip = get_ipython()
        ip.run_cell('a = 10')
        ip.run_cell('def f(x):\n    return x + a')
        ip.run_cell('b = f(12)')
        self.assertEqual(ip.user_ns['b'], 22)

    def test_bad_custom_tb(self):
        if False:
            while True:
                i = 10
        'Check that InteractiveShell is protected from bad custom exception handlers'
        ip.set_custom_exc((IOError,), lambda etype, value, tb: 1 / 0)
        self.assertEqual(ip.custom_exceptions, (IOError,))
        with tt.AssertPrints('Custom TB Handler failed', channel='stderr'):
            ip.run_cell(u'raise IOError("foo")')
        self.assertEqual(ip.custom_exceptions, ())

    def test_bad_custom_tb_return(self):
        if False:
            while True:
                i = 10
        'Check that InteractiveShell is protected from bad return types in custom exception handlers'
        ip.set_custom_exc((NameError,), lambda etype, value, tb, tb_offset=None: 1)
        self.assertEqual(ip.custom_exceptions, (NameError,))
        with tt.AssertPrints('Custom TB Handler failed', channel='stderr'):
            ip.run_cell(u'a=abracadabra')
        self.assertEqual(ip.custom_exceptions, ())

    def test_drop_by_id(self):
        if False:
            print('Hello World!')
        myvars = {'a': object(), 'b': object(), 'c': object()}
        ip.push(myvars, interactive=False)
        for name in myvars:
            assert name in ip.user_ns, name
            assert name in ip.user_ns_hidden, name
        ip.user_ns['b'] = 12
        ip.drop_by_id(myvars)
        for name in ['a', 'c']:
            assert name not in ip.user_ns, name
            assert name not in ip.user_ns_hidden, name
        assert ip.user_ns['b'] == 12
        ip.reset()

    def test_var_expand(self):
        if False:
            for i in range(10):
                print('nop')
        ip.user_ns['f'] = u'Caño'
        self.assertEqual(ip.var_expand(u'echo $f'), u'echo Caño')
        self.assertEqual(ip.var_expand(u'echo {f}'), u'echo Caño')
        self.assertEqual(ip.var_expand(u'echo {f[:-1]}'), u'echo Cañ')
        self.assertEqual(ip.var_expand(u'echo {1*2}'), u'echo 2')
        self.assertEqual(ip.var_expand(u"grep x | awk '{print $1}'"), u"grep x | awk '{print $1}'")
        ip.user_ns['f'] = b'Ca\xc3\xb1o'
        ip.var_expand(u'echo $f')

    def test_var_expand_local(self):
        if False:
            print('Hello World!')
        'Test local variable expansion in !system and %magic calls'
        ip.run_cell('def test():\n    lvar = "ttt"\n    ret = !echo {lvar}\n    return ret[0]\n')
        res = ip.user_ns['test']()
        self.assertIn('ttt', res)
        ip.run_cell('def makemacro():\n    macroname = "macro_var_expand_locals"\n    %macro {macroname} codestr\n')
        ip.user_ns['codestr'] = 'str(12)'
        ip.run_cell('makemacro()')
        self.assertIn('macro_var_expand_locals', ip.user_ns)

    def test_var_expand_self(self):
        if False:
            while True:
                i = 10
        "Test variable expansion with the name 'self', which was failing.\n        \n        See https://github.com/ipython/ipython/issues/1878#issuecomment-7698218\n        "
        ip.run_cell('class cTest:\n  classvar="see me"\n  def test(self):\n    res = !echo Variable: {self.classvar}\n    return res[0]\n')
        self.assertIn('see me', ip.user_ns['cTest']().test())

    def test_bad_var_expand(self):
        if False:
            for i in range(10):
                print('nop')
        "var_expand on invalid formats shouldn't raise"
        self.assertEqual(ip.var_expand(u"{'a':5}"), u"{'a':5}")
        self.assertEqual(ip.var_expand(u'{asdf}'), u'{asdf}')
        self.assertEqual(ip.var_expand(u'{1/0}'), u'{1/0}')

    def test_silent_postexec(self):
        if False:
            print('Hello World!')
        "run_cell(silent=True) doesn't invoke pre/post_run_cell callbacks"
        pre_explicit = mock.Mock()
        pre_always = mock.Mock()
        post_explicit = mock.Mock()
        post_always = mock.Mock()
        all_mocks = [pre_explicit, pre_always, post_explicit, post_always]
        ip.events.register('pre_run_cell', pre_explicit)
        ip.events.register('pre_execute', pre_always)
        ip.events.register('post_run_cell', post_explicit)
        ip.events.register('post_execute', post_always)
        try:
            ip.run_cell('1', silent=True)
            assert pre_always.called
            assert not pre_explicit.called
            assert post_always.called
            assert not post_explicit.called
            ip.run_cell('1')
            assert pre_explicit.called
            assert post_explicit.called
            (info,) = pre_explicit.call_args[0]
            (result,) = post_explicit.call_args[0]
            self.assertEqual(info, result.info)
            [m.reset_mock() for m in all_mocks]
            ip.run_cell('syntax error')
            assert pre_always.called
            assert pre_explicit.called
            assert post_always.called
            assert post_explicit.called
            (info,) = pre_explicit.call_args[0]
            (result,) = post_explicit.call_args[0]
            self.assertEqual(info, result.info)
        finally:
            ip.events.unregister('pre_run_cell', pre_explicit)
            ip.events.unregister('pre_execute', pre_always)
            ip.events.unregister('post_run_cell', post_explicit)
            ip.events.unregister('post_execute', post_always)

    def test_silent_noadvance(self):
        if False:
            i = 10
            return i + 15
        "run_cell(silent=True) doesn't advance execution_count"
        ec = ip.execution_count
        ip.run_cell('1', store_history=True, silent=True)
        self.assertEqual(ec, ip.execution_count)
        ip.run_cell('1', store_history=True)
        self.assertEqual(ec + 1, ip.execution_count)

    def test_silent_nodisplayhook(self):
        if False:
            for i in range(10):
                print('nop')
        "run_cell(silent=True) doesn't trigger displayhook"
        d = dict(called=False)
        trap = ip.display_trap
        save_hook = trap.hook

        def failing_hook(*args, **kwargs):
            if False:
                while True:
                    i = 10
            d['called'] = True
        try:
            trap.hook = failing_hook
            res = ip.run_cell('1', silent=True)
            self.assertFalse(d['called'])
            self.assertIsNone(res.result)
            ip.run_cell('1')
            self.assertTrue(d['called'])
        finally:
            trap.hook = save_hook

    def test_ofind_line_magic(self):
        if False:
            for i in range(10):
                print('nop')
        from IPython.core.magic import register_line_magic

        @register_line_magic
        def lmagic(line):
            if False:
                while True:
                    i = 10
            'A line magic'
        lfind = ip._ofind('lmagic')
        info = OInfo(found=True, isalias=False, ismagic=True, namespace='IPython internal', obj=lmagic, parent=None)
        self.assertEqual(lfind, info)

    def test_ofind_cell_magic(self):
        if False:
            while True:
                i = 10
        from IPython.core.magic import register_cell_magic

        @register_cell_magic
        def cmagic(line, cell):
            if False:
                while True:
                    i = 10
            'A cell magic'
        find = ip._ofind('cmagic')
        info = OInfo(found=True, isalias=False, ismagic=True, namespace='IPython internal', obj=cmagic, parent=None)
        self.assertEqual(find, info)

    def test_ofind_property_with_error(self):
        if False:
            for i in range(10):
                print('nop')

        class A(object):

            @property
            def foo(self):
                if False:
                    while True:
                        i = 10
                raise NotImplementedError()
        a = A()
        found = ip._ofind('a.foo', [('locals', locals())])
        info = OInfo(found=True, isalias=False, ismagic=False, namespace='locals', obj=A.foo, parent=a)
        self.assertEqual(found, info)

    def test_ofind_multiple_attribute_lookups(self):
        if False:
            print('Hello World!')

        class A(object):

            @property
            def foo(self):
                if False:
                    print('Hello World!')
                raise NotImplementedError()
        a = A()
        a.a = A()
        a.a.a = A()
        found = ip._ofind('a.a.a.foo', [('locals', locals())])
        info = OInfo(found=True, isalias=False, ismagic=False, namespace='locals', obj=A.foo, parent=a.a.a)
        self.assertEqual(found, info)

    def test_ofind_slotted_attributes(self):
        if False:
            while True:
                i = 10

        class A(object):
            __slots__ = ['foo']

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.foo = 'bar'
        a = A()
        found = ip._ofind('a.foo', [('locals', locals())])
        info = OInfo(found=True, isalias=False, ismagic=False, namespace='locals', obj=a.foo, parent=a)
        self.assertEqual(found, info)
        found = ip._ofind('a.bar', [('locals', locals())])
        expected = OInfo(found=False, isalias=False, ismagic=False, namespace=None, obj=None, parent=a)
        assert found == expected

    def test_ofind_prefers_property_to_instance_level_attribute(self):
        if False:
            i = 10
            return i + 15

        class A(object):

            @property
            def foo(self):
                if False:
                    print('Hello World!')
                return 'bar'
        a = A()
        a.__dict__['foo'] = 'baz'
        self.assertEqual(a.foo, 'bar')
        found = ip._ofind('a.foo', [('locals', locals())])
        self.assertIs(found.obj, A.foo)

    def test_custom_syntaxerror_exception(self):
        if False:
            while True:
                i = 10
        called = []

        def my_handler(shell, etype, value, tb, tb_offset=None):
            if False:
                return 10
            called.append(etype)
            shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
        ip.set_custom_exc((SyntaxError,), my_handler)
        try:
            ip.run_cell('1f')
            self.assertEqual(called, [SyntaxError])
        finally:
            ip.set_custom_exc((), None)

    def test_custom_exception(self):
        if False:
            i = 10
            return i + 15
        called = []

        def my_handler(shell, etype, value, tb, tb_offset=None):
            if False:
                return 10
            called.append(etype)
            shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
        ip.set_custom_exc((ValueError,), my_handler)
        try:
            res = ip.run_cell("raise ValueError('test')")
            self.assertEqual(called, [ValueError])
            self.assertIsInstance(res.error_in_exec, ValueError)
        finally:
            ip.set_custom_exc((), None)

    @mock.patch('builtins.print')
    def test_showtraceback_with_surrogates(self, mocked_print):
        if False:
            i = 10
            return i + 15
        values = []

        def mock_print_func(value, sep=' ', end='\n', file=sys.stdout, flush=False):
            if False:
                while True:
                    i = 10
            values.append(value)
            if value == chr(55551):
                raise UnicodeEncodeError('utf-8', chr(55551), 0, 1, '')
        mocked_print.side_effect = mock_print_func
        interactiveshell.InteractiveShell._showtraceback(ip, None, None, chr(55551))
        self.assertEqual(mocked_print.call_count, 2)
        self.assertEqual(values, [chr(55551), '\\ud8ff'])

    def test_mktempfile(self):
        if False:
            while True:
                i = 10
        filename = ip.mktempfile()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('abc')
        filename = ip.mktempfile(data='blah')
        with open(filename, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), 'blah')

    def test_new_main_mod(self):
        if False:
            print('Hello World!')
        name = u'jiefmw'
        mod = ip.new_main_mod(u'%s.py' % name, name)
        self.assertEqual(mod.__name__, name)

    def test_get_exception_only(self):
        if False:
            i = 10
            return i + 15
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            msg = ip.get_exception_only()
        self.assertEqual(msg, 'KeyboardInterrupt\n')
        try:
            raise DerivedInterrupt('foo')
        except KeyboardInterrupt:
            msg = ip.get_exception_only()
        self.assertEqual(msg, 'IPython.core.tests.test_interactiveshell.DerivedInterrupt: foo\n')

    def test_inspect_text(self):
        if False:
            while True:
                i = 10
        ip.run_cell('a = 5')
        text = ip.object_inspect_text('a')
        self.assertIsInstance(text, str)

    def test_last_execution_result(self):
        if False:
            return 10
        ' Check that last execution result gets set correctly (GH-10702) '
        result = ip.run_cell('a = 5; a')
        self.assertTrue(ip.last_execution_succeeded)
        self.assertEqual(ip.last_execution_result.result, 5)
        result = ip.run_cell('a = x_invalid_id_x')
        self.assertFalse(ip.last_execution_succeeded)
        self.assertFalse(ip.last_execution_result.success)
        self.assertIsInstance(ip.last_execution_result.error_in_exec, NameError)

    def test_reset_aliasing(self):
        if False:
            i = 10
            return i + 15
        ' Check that standard posix aliases work after %reset. '
        if os.name != 'posix':
            return
        ip.reset()
        for cmd in ('clear', 'more', 'less', 'man'):
            res = ip.run_cell('%' + cmd)
            self.assertEqual(res.success, True)

class TestSafeExecfileNonAsciiPath(unittest.TestCase):

    @onlyif_unicode_paths
    def setUp(self):
        if False:
            return 10
        self.BASETESTDIR = tempfile.mkdtemp()
        self.TESTDIR = join(self.BASETESTDIR, u'åäö')
        os.mkdir(self.TESTDIR)
        with open(join(self.TESTDIR, 'åäötestscript.py'), 'w', encoding='utf-8') as sfile:
            sfile.write('pass\n')
        self.oldpath = os.getcwd()
        os.chdir(self.TESTDIR)
        self.fname = u'åäötestscript.py'

    def tearDown(self):
        if False:
            print('Hello World!')
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test safe_execfile with non-ascii path\n        '
        ip.safe_execfile(self.fname, {}, raise_exceptions=True)

class ExitCodeChecks(tt.TempFileMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.system = ip.system_raw

    def test_exit_code_ok(self):
        if False:
            for i in range(10):
                print('nop')
        self.system('exit 0')
        self.assertEqual(ip.user_ns['_exit_code'], 0)

    def test_exit_code_error(self):
        if False:
            return 10
        self.system('exit 1')
        self.assertEqual(ip.user_ns['_exit_code'], 1)

    @skipif(not hasattr(signal, 'SIGALRM'))
    def test_exit_code_signal(self):
        if False:
            i = 10
            return i + 15
        self.mktmp('import signal, time\nsignal.setitimer(signal.ITIMER_REAL, 0.1)\ntime.sleep(1)\n')
        self.system('%s %s' % (sys.executable, self.fname))
        self.assertEqual(ip.user_ns['_exit_code'], -signal.SIGALRM)

    @onlyif_cmds_exist('csh')
    def test_exit_code_signal_csh(self):
        if False:
            while True:
                i = 10
        SHELL = os.environ.get('SHELL', None)
        os.environ['SHELL'] = find_cmd('csh')
        try:
            self.test_exit_code_signal()
        finally:
            if SHELL is not None:
                os.environ['SHELL'] = SHELL
            else:
                del os.environ['SHELL']

class TestSystemRaw(ExitCodeChecks):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.system = ip.system_raw

    @onlyif_unicode_paths
    def test_1(self):
        if False:
            while True:
                i = 10
        'Test system_raw with non-ascii cmd\n        '
        cmd = u'python -c "\'åäö\'"   '
        ip.system_raw(cmd)

    @mock.patch('subprocess.call', side_effect=KeyboardInterrupt)
    @mock.patch('os.system', side_effect=KeyboardInterrupt)
    def test_control_c(self, *mocks):
        if False:
            while True:
                i = 10
        try:
            self.system('sleep 1 # wont happen')
        except KeyboardInterrupt:
            self.fail('system call should intercept keyboard interrupt from subprocess.call')
        self.assertEqual(ip.user_ns['_exit_code'], -signal.SIGINT)

@pytest.mark.parametrize('magic_cmd', ['pip', 'conda', 'cd'])
def test_magic_warnings(magic_cmd):
    if False:
        while True:
            i = 10
    if sys.platform == 'win32':
        to_mock = 'os.system'
        (expected_arg, expected_kwargs) = (magic_cmd, dict())
    else:
        to_mock = 'subprocess.call'
        (expected_arg, expected_kwargs) = (magic_cmd, dict(shell=True, executable=os.environ.get('SHELL', None)))
    with mock.patch(to_mock, return_value=0) as mock_sub:
        with pytest.warns(Warning, match='You executed the system command'):
            ip.system_raw(magic_cmd)
        mock_sub.assert_called_once_with(expected_arg, **expected_kwargs)

class TestSystemPipedExitCode(ExitCodeChecks):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.system = ip.system_piped

    @skip_win32
    def test_exit_code_ok(self):
        if False:
            for i in range(10):
                print('nop')
        ExitCodeChecks.test_exit_code_ok(self)

    @skip_win32
    def test_exit_code_error(self):
        if False:
            i = 10
            return i + 15
        ExitCodeChecks.test_exit_code_error(self)

    @skip_win32
    def test_exit_code_signal(self):
        if False:
            return 10
        ExitCodeChecks.test_exit_code_signal(self)

class TestModules(tt.TempFileMixin):

    def test_extraneous_loads(self):
        if False:
            i = 10
            return i + 15
        "Test we're not loading modules on startup that we shouldn't.\n        "
        self.mktmp("import sys\nprint('numpy' in sys.modules)\nprint('ipyparallel' in sys.modules)\nprint('ipykernel' in sys.modules)\n")
        out = 'False\nFalse\nFalse\n'
        tt.ipexec_validate(self.fname, out)

class Negator(ast.NodeTransformer):
    """Negates all number literals in an AST."""

    def visit_Num(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.n = -node.n
        return node

    def visit_Constant(self, node):
        if False:
            return 10
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node

class TestAstTransform(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.negator = Negator()
        ip.ast_transformers.append(self.negator)

    def tearDown(self):
        if False:
            print('Hello World!')
        ip.ast_transformers.remove(self.negator)

    def test_non_int_const(self):
        if False:
            print('Hello World!')
        with tt.AssertPrints('hello'):
            ip.run_cell('print("hello")')

    def test_run_cell(self):
        if False:
            for i in range(10):
                print('nop')
        with tt.AssertPrints('-34'):
            ip.run_cell('print(12 + 22)')
        ip.user_ns['n'] = 55
        with tt.AssertNotPrints('-55'):
            ip.run_cell('print(n)')

    def test_timeit(self):
        if False:
            for i in range(10):
                print('nop')
        called = set()

        def f(x):
            if False:
                print('Hello World!')
            called.add(x)
        ip.push({'f': f})
        with tt.AssertPrints('std. dev. of'):
            ip.run_line_magic('timeit', '-n1 f(1)')
        self.assertEqual(called, {-1})
        called.clear()
        with tt.AssertPrints('std. dev. of'):
            ip.run_cell_magic('timeit', '-n1 f(2)', 'f(3)')
        self.assertEqual(called, {-2, -3})

    def test_time(self):
        if False:
            print('Hello World!')
        called = []

        def f(x):
            if False:
                return 10
            called.append(x)
        ip.push({'f': f})
        with tt.AssertPrints('Wall time: '):
            ip.run_line_magic('time', 'f(5+9)')
        self.assertEqual(called, [-14])
        called[:] = []
        with tt.AssertPrints('Wall time: '):
            ip.run_line_magic('time', 'a = f(-3 + -2)')
        self.assertEqual(called, [5])

    def test_macro(self):
        if False:
            return 10
        ip.push({'a': 10})
        ip.define_macro('amacro', 'a+=1\nprint(a)')
        with tt.AssertPrints('9'):
            ip.run_cell('amacro')
        with tt.AssertPrints('8'):
            ip.run_cell('amacro')

class TestMiscTransform(unittest.TestCase):

    def test_transform_only_once(self):
        if False:
            print('Hello World!')
        cleanup = 0
        line_t = 0

        def count_cleanup(lines):
            if False:
                return 10
            nonlocal cleanup
            cleanup += 1
            return lines

        def count_line_t(lines):
            if False:
                return 10
            nonlocal line_t
            line_t += 1
            return lines
        ip.input_transformer_manager.cleanup_transforms.append(count_cleanup)
        ip.input_transformer_manager.line_transforms.append(count_line_t)
        ip.run_cell('1')
        assert cleanup == 1
        assert line_t == 1

class IntegerWrapper(ast.NodeTransformer):
    """Wraps all integers in a call to Integer()"""

    def visit_Num(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node.n, int):
            return ast.Call(func=ast.Name(id='Integer', ctx=ast.Load()), args=[node], keywords=[])
        return node

    def visit_Constant(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node

class TestAstTransform2(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.intwrapper = IntegerWrapper()
        ip.ast_transformers.append(self.intwrapper)
        self.calls = []

        def Integer(*args):
            if False:
                for i in range(10):
                    print('nop')
            self.calls.append(args)
            return args
        ip.push({'Integer': Integer})

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ip.ast_transformers.remove(self.intwrapper)
        del ip.user_ns['Integer']

    def test_run_cell(self):
        if False:
            print('Hello World!')
        ip.run_cell('n = 2')
        self.assertEqual(self.calls, [(2,)])
        ip.run_cell('o = 2.0')
        self.assertEqual(ip.user_ns['o'], 2.0)

    def test_run_cell_non_int(self):
        if False:
            while True:
                i = 10
        ip.run_cell("n = 'a'")
        assert self.calls == []

    def test_timeit(self):
        if False:
            print('Hello World!')
        called = set()

        def f(x):
            if False:
                print('Hello World!')
            called.add(x)
        ip.push({'f': f})
        with tt.AssertPrints('std. dev. of'):
            ip.run_line_magic('timeit', '-n1 f(1)')
        self.assertEqual(called, {(1,)})
        called.clear()
        with tt.AssertPrints('std. dev. of'):
            ip.run_cell_magic('timeit', '-n1 f(2)', 'f(3)')
        self.assertEqual(called, {(2,), (3,)})

class ErrorTransformer(ast.NodeTransformer):
    """Throws an error when it sees a number."""

    def visit_Constant(self, node):
        if False:
            print('Hello World!')
        if isinstance(node.value, int):
            raise ValueError('test')
        return node

class TestAstTransformError(unittest.TestCase):

    def test_unregistering(self):
        if False:
            for i in range(10):
                print('nop')
        err_transformer = ErrorTransformer()
        ip.ast_transformers.append(err_transformer)
        with self.assertWarnsRegex(UserWarning, 'It will be unregistered'):
            ip.run_cell('1 + 2')
        self.assertNotIn(err_transformer, ip.ast_transformers)

class StringRejector(ast.NodeTransformer):
    """Throws an InputRejected when it sees a string literal.

    Used to verify that NodeTransformers can signal that a piece of code should
    not be executed by throwing an InputRejected.
    """

    def visit_Constant(self, node):
        if False:
            print('Hello World!')
        if isinstance(node.value, str):
            raise InputRejected('test')
        return node

class TestAstTransformInputRejection(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.transformer = StringRejector()
        ip.ast_transformers.append(self.transformer)

    def tearDown(self):
        if False:
            return 10
        ip.ast_transformers.remove(self.transformer)

    def test_input_rejection(self):
        if False:
            i = 10
            return i + 15
        'Check that NodeTransformers can reject input.'
        expect_exception_tb = tt.AssertPrints('InputRejected: test')
        expect_no_cell_output = tt.AssertNotPrints("'unsafe'", suppress=False)
        with expect_exception_tb, expect_no_cell_output:
            ip.run_cell("'unsafe'")
        with expect_exception_tb, expect_no_cell_output:
            res = ip.run_cell("'unsafe'")
        self.assertIsInstance(res.error_before_exec, InputRejected)

def test__IPYTHON__():
    if False:
        for i in range(10):
            print('nop')
    __IPYTHON__

class DummyRepr(object):

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'DummyRepr'

    def _repr_html_(self):
        if False:
            while True:
                i = 10
        return '<b>dummy</b>'

    def _repr_javascript_(self):
        if False:
            i = 10
            return i + 15
        return ("console.log('hi');", {'key': 'value'})

def test_user_variables():
    if False:
        while True:
            i = 10
    ip.display_formatter.active_types = ip.display_formatter.format_types
    ip.user_ns['dummy'] = d = DummyRepr()
    keys = {'dummy', 'doesnotexist'}
    r = ip.user_expressions({key: key for key in keys})
    assert keys == set(r.keys())
    dummy = r['dummy']
    assert {'status', 'data', 'metadata'} == set(dummy.keys())
    assert dummy['status'] == 'ok'
    data = dummy['data']
    metadata = dummy['metadata']
    assert data.get('text/html') == d._repr_html_()
    (js, jsmd) = d._repr_javascript_()
    assert data.get('application/javascript') == js
    assert metadata.get('application/javascript') == jsmd
    dne = r['doesnotexist']
    assert dne['status'] == 'error'
    assert dne['ename'] == 'NameError'
    ip.display_formatter.active_types = ['text/plain']

def test_user_expression():
    if False:
        print('Hello World!')
    ip.display_formatter.active_types = ip.display_formatter.format_types
    query = {'a': '1 + 2', 'b': '1/0'}
    r = ip.user_expressions(query)
    import pprint
    pprint.pprint(r)
    assert set(r.keys()) == set(query.keys())
    a = r['a']
    assert {'status', 'data', 'metadata'} == set(a.keys())
    assert a['status'] == 'ok'
    data = a['data']
    metadata = a['metadata']
    assert data.get('text/plain') == '3'
    b = r['b']
    assert b['status'] == 'error'
    assert b['ename'] == 'ZeroDivisionError'
    ip.display_formatter.active_types = ['text/plain']

class TestSyntaxErrorTransformer(unittest.TestCase):
    """Check that SyntaxError raised by an input transformer is handled by run_cell()"""

    @staticmethod
    def transformer(lines):
        if False:
            for i in range(10):
                print('nop')
        for line in lines:
            pos = line.find('syntaxerror')
            if pos >= 0:
                e = SyntaxError('input contains "syntaxerror"')
                e.text = line
                e.offset = pos + 1
                raise e
        return lines

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ip.input_transformers_post.append(self.transformer)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ip.input_transformers_post.remove(self.transformer)

    def test_syntaxerror_input_transformer(self):
        if False:
            print('Hello World!')
        with tt.AssertPrints('1234'):
            ip.run_cell('1234')
        with tt.AssertPrints('SyntaxError: invalid syntax'):
            ip.run_cell('1 2 3')
        with tt.AssertPrints('SyntaxError: input contains "syntaxerror"'):
            ip.run_cell('2345  # syntaxerror')
        with tt.AssertPrints('3456'):
            ip.run_cell('3456')

class TestWarningSuppression(unittest.TestCase):

    def test_warning_suppression(self):
        if False:
            i = 10
            return i + 15
        ip.run_cell('import warnings')
        try:
            with self.assertWarnsRegex(UserWarning, 'asdf'):
                ip.run_cell("warnings.warn('asdf')")
            with self.assertWarnsRegex(UserWarning, 'asdf'):
                ip.run_cell("warnings.warn('asdf')")
        finally:
            ip.run_cell('del warnings')

    def test_deprecation_warning(self):
        if False:
            print('Hello World!')
        ip.run_cell('\nimport warnings\ndef wrn():\n    warnings.warn(\n        "I AM  A WARNING",\n        DeprecationWarning\n    )\n        ')
        try:
            with self.assertWarnsRegex(DeprecationWarning, 'I AM  A WARNING'):
                ip.run_cell('wrn()')
        finally:
            ip.run_cell('del warnings')
            ip.run_cell('del wrn')

class TestImportNoDeprecate(tt.TempFileMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        'Make a valid python temp file.'
        self.mktmp('\nimport warnings\ndef wrn():\n    warnings.warn(\n        "I AM  A WARNING",\n        DeprecationWarning\n    )\n')
        super().setUp()

    def test_no_dep(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        No deprecation warning should be raised from imported functions\n        '
        ip.run_cell('from {} import wrn'.format(self.fname))
        with tt.AssertNotPrints('I AM  A WARNING'):
            ip.run_cell('wrn()')
        ip.run_cell('del wrn')

def test_custom_exc_count():
    if False:
        return 10
    hook = mock.Mock(return_value=None)
    ip.set_custom_exc((SyntaxError,), hook)
    before = ip.execution_count
    ip.run_cell('def foo()', store_history=True)
    ip.set_custom_exc((), None)
    assert hook.call_count == 1
    assert ip.execution_count == before + 1

def test_run_cell_async():
    if False:
        print('Hello World!')
    ip.run_cell('import asyncio')
    coro = ip.run_cell_async('await asyncio.sleep(0.01)\n5')
    assert asyncio.iscoroutine(coro)
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(coro)
    assert isinstance(result, interactiveshell.ExecutionResult)
    assert result.result == 5

def test_run_cell_await():
    if False:
        return 10
    ip.run_cell('import asyncio')
    result = ip.run_cell('await asyncio.sleep(0.01); 10')
    assert ip.user_ns['_'] == 10

def test_run_cell_asyncio_run():
    if False:
        print('Hello World!')
    ip.run_cell('import asyncio')
    result = ip.run_cell('await asyncio.sleep(0.01); 1')
    assert ip.user_ns['_'] == 1
    result = ip.run_cell('asyncio.run(asyncio.sleep(0.01)); 2')
    assert ip.user_ns['_'] == 2
    result = ip.run_cell('await asyncio.sleep(0.01); 3')
    assert ip.user_ns['_'] == 3

def test_should_run_async():
    if False:
        while True:
            i = 10
    assert not ip.should_run_async('a = 5', transformed_cell='a = 5')
    assert ip.should_run_async('await x', transformed_cell='await x')
    assert ip.should_run_async('import asyncio; await asyncio.sleep(1)', transformed_cell='import asyncio; await asyncio.sleep(1)')

def test_set_custom_completer():
    if False:
        print('Hello World!')
    num_completers = len(ip.Completer.matchers)

    def foo(*args, **kwargs):
        if False:
            return 10
        return "I'm a completer!"
    ip.set_custom_completer(foo, 0)
    assert len(ip.Completer.matchers) == num_completers + 1
    assert ip.Completer.matchers[0]() == "I'm a completer!"
    ip.Completer.custom_matchers.pop()

class TestShowTracebackAttack(unittest.TestCase):
    """Test that the interactive shell is resilient against the client attack of
    manipulating the showtracebacks method. These attacks shouldn't result in an
    unhandled exception in the kernel."""

    def setUp(self):
        if False:
            return 10
        self.orig_showtraceback = interactiveshell.InteractiveShell.showtraceback

    def tearDown(self):
        if False:
            while True:
                i = 10
        interactiveshell.InteractiveShell.showtraceback = self.orig_showtraceback

    def test_set_show_tracebacks_none(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the case of the client setting showtracebacks to None'
        result = ip.run_cell('\n            import IPython.core.interactiveshell\n            IPython.core.interactiveshell.InteractiveShell.showtraceback = None\n\n            assert False, "This should not raise an exception"\n        ')
        print(result)
        assert result.result is None
        assert isinstance(result.error_in_exec, TypeError)
        assert str(result.error_in_exec) == "'NoneType' object is not callable"

    def test_set_show_tracebacks_noop(self):
        if False:
            return 10
        'Test the case of the client setting showtracebacks to a no op lambda'
        result = ip.run_cell('\n            import IPython.core.interactiveshell\n            IPython.core.interactiveshell.InteractiveShell.showtraceback = lambda *args, **kwargs: None\n\n            assert False, "This should not raise an exception"\n        ')
        print(result)
        assert result.result is None
        assert isinstance(result.error_in_exec, AssertionError)
        assert str(result.error_in_exec) == 'This should not raise an exception'