import os
import platform
import re
import subprocess
import sys
import sysconfig
import textwrap
import unittest
from test import support
from test.support import findfile, python_is_optimized

def get_gdb_version():
    if False:
        i = 10
        return i + 15
    try:
        cmd = ['gdb', '-nx', '--version']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        with proc:
            (version, stderr) = proc.communicate()
        if proc.returncode:
            raise Exception(f"Command {' '.join(cmd)!r} failed with exit code {proc.returncode}: stdout={version!r} stderr={stderr!r}")
    except OSError:
        raise unittest.SkipTest("Couldn't find gdb on the path")
    match = re.search('^(?:GNU|HP) gdb.*?\\b(\\d+)\\.(\\d+)', version)
    if match is None:
        raise Exception('unable to parse GDB version: %r' % version)
    return (version, int(match.group(1)), int(match.group(2)))
(gdb_version, gdb_major_version, gdb_minor_version) = get_gdb_version()
if gdb_major_version < 7:
    raise unittest.SkipTest("gdb versions before 7.0 didn't support python embedding. Saw %s.%s:\n%s" % (gdb_major_version, gdb_minor_version, gdb_version))
if not sysconfig.is_python_build():
    raise unittest.SkipTest('test_gdb only works on source builds at the moment.')
if 'Clang' in platform.python_compiler() and sys.platform == 'darwin':
    raise unittest.SkipTest("test_gdb doesn't work correctly when python is built with LLVM clang")
if (sysconfig.get_config_var('PGO_PROF_USE_FLAG') or 'xxx') in (sysconfig.get_config_var('PY_CORE_CFLAGS') or ''):
    raise unittest.SkipTest('test_gdb is not reliable on PGO builds')
checkout_hook_path = os.path.join(os.path.dirname(sys.executable), 'python-gdb.py')
PYTHONHASHSEED = '123'

def cet_protection():
    if False:
        while True:
            i = 10
    cflags = sysconfig.get_config_var('CFLAGS')
    if not cflags:
        return False
    flags = cflags.split()
    return '-mcet' in flags and any((flag.startswith('-fcf-protection') and (not flag.endswith(('=none', '=return'))) for flag in flags))
CET_PROTECTION = cet_protection()

def run_gdb(*args, **env_vars):
    if False:
        for i in range(10):
            print('nop')
    'Runs gdb in --batch mode with the additional arguments given by *args.\n\n    Returns its (stdout, stderr) decoded from utf-8 using the replace handler.\n    '
    if env_vars:
        env = os.environ.copy()
        env.update(env_vars)
    else:
        env = None
    base_cmd = ('gdb', '--batch', '-nx')
    if (gdb_major_version, gdb_minor_version) >= (7, 4):
        base_cmd += ('-iex', 'add-auto-load-safe-path ' + checkout_hook_path)
    proc = subprocess.Popen(base_cmd + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    with proc:
        (out, err) = proc.communicate()
    return (out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace'))
(gdbpy_version, _) = run_gdb('--eval-command=python import sys; print(sys.version_info)')
if not gdbpy_version:
    raise unittest.SkipTest('gdb not built with embedded python support')
(_, gdbpy_errors) = run_gdb('--args', sys.executable)
if 'auto-loading has been declined' in gdbpy_errors:
    msg = 'gdb security settings prevent use of custom hooks: '
    raise unittest.SkipTest(msg + gdbpy_errors.rstrip())

def gdb_has_frame_select():
    if False:
        i = 10
        return i + 15
    (stdout, _) = run_gdb('--eval-command=python print(dir(gdb.Frame))')
    m = re.match('.*\\[(.*)\\].*', stdout)
    if not m:
        raise unittest.SkipTest('Unable to parse output from gdb.Frame.select test')
    gdb_frame_dir = m.group(1).split(', ')
    return "'select'" in gdb_frame_dir
HAS_PYUP_PYDOWN = gdb_has_frame_select()
BREAKPOINT_FN = 'builtin_id'

@unittest.skipIf(support.PGO, 'not useful for PGO')
class DebuggerTests(unittest.TestCase):
    """Test that the debugger can debug Python."""

    def get_stack_trace(self, source=None, script=None, breakpoint=BREAKPOINT_FN, cmds_after_breakpoint=None, import_site=False, ignore_stderr=False):
        if False:
            while True:
                i = 10
        "\n        Run 'python -c SOURCE' under gdb with a breakpoint.\n\n        Support injecting commands after the breakpoint is reached\n\n        Returns the stdout from gdb\n\n        cmds_after_breakpoint: if provided, a list of strings: gdb commands\n        "
        commands = ['set breakpoint pending yes', 'break %s' % breakpoint, 'set print address off', 'run']
        if (gdb_major_version, gdb_minor_version) >= (7, 4):
            commands += ['set print entry-values no']
        if cmds_after_breakpoint:
            if CET_PROTECTION:
                commands += ['next']
            commands += cmds_after_breakpoint
        else:
            commands += ['backtrace']
        args = ['--eval-command=%s' % cmd for cmd in commands]
        args += ['--args', sys.executable]
        args.extend(subprocess._args_from_interpreter_flags())
        if not import_site:
            args += ['-S']
        if source:
            args += ['-c', source]
        elif script:
            args += [script]
        (out, err) = run_gdb(*args, PYTHONHASHSEED=PYTHONHASHSEED)
        if not ignore_stderr:
            for line in err.splitlines():
                print(line, file=sys.stderr)
        if 'PC not saved' in err:
            raise unittest.SkipTest('gdb cannot walk the frame object because the Program Counter is not present')
        for pattern in ('(frame information optimized out)', 'Unable to read information on python frame'):
            if pattern in out:
                raise unittest.SkipTest(f'{pattern!r} found in gdb output')
        return out

    def get_gdb_repr(self, source, cmds_after_breakpoint=None, import_site=False):
        if False:
            while True:
                i = 10
        cmds_after_breakpoint = cmds_after_breakpoint or ['backtrace 1']
        gdb_output = self.get_stack_trace(source, breakpoint=BREAKPOINT_FN, cmds_after_breakpoint=cmds_after_breakpoint, import_site=import_site)
        m = re.search('#0\\s+builtin_id\\s+\\(self\\=.*,\\s+v=\\s*(.*?)?\\)\\s+at\\s+\\S*[A-Za-z]+/[A-Za-z0-9_-]+\\.c', gdb_output, re.DOTALL)
        if not m:
            self.fail('Unexpected gdb output: %r\n%s' % (gdb_output, gdb_output))
        return (m.group(1), gdb_output)

    def assertEndsWith(self, actual, exp_end):
        if False:
            return 10
        'Ensure that the given "actual" string ends with "exp_end"'
        self.assertTrue(actual.endswith(exp_end), msg='%r did not end with %r' % (actual, exp_end))

    def assertMultilineMatches(self, actual, pattern):
        if False:
            print('Hello World!')
        m = re.match(pattern, actual, re.DOTALL)
        if not m:
            self.fail(msg='%r did not match %r' % (actual, pattern))

    def get_sample_script(self):
        if False:
            i = 10
            return i + 15
        return findfile('gdb_sample.py')

class PrettyPrintTests(DebuggerTests):

    def test_getting_backtrace(self):
        if False:
            return 10
        gdb_output = self.get_stack_trace('id(42)')
        self.assertTrue(BREAKPOINT_FN in gdb_output)

    def assertGdbRepr(self, val, exp_repr=None):
        if False:
            print('Hello World!')
        (gdb_repr, gdb_output) = self.get_gdb_repr('id(' + ascii(val) + ')')
        if not exp_repr:
            exp_repr = repr(val)
        self.assertEqual(gdb_repr, exp_repr, '%r did not equal expected %r; full output was:\n%s' % (gdb_repr, exp_repr, gdb_output))

    def test_int(self):
        if False:
            i = 10
            return i + 15
        'Verify the pretty-printing of various int values'
        self.assertGdbRepr(42)
        self.assertGdbRepr(0)
        self.assertGdbRepr(-7)
        self.assertGdbRepr(1000000000000)
        self.assertGdbRepr(-1000000000000000)

    def test_singletons(self):
        if False:
            while True:
                i = 10
        'Verify the pretty-printing of True, False and None'
        self.assertGdbRepr(True)
        self.assertGdbRepr(False)
        self.assertGdbRepr(None)

    def test_dicts(self):
        if False:
            return 10
        'Verify the pretty-printing of dictionaries'
        self.assertGdbRepr({})
        self.assertGdbRepr({'foo': 'bar'}, "{'foo': 'bar'}")
        self.assertGdbRepr({'foo': 'bar', 'douglas': 42}, "{'foo': 'bar', 'douglas': 42}")

    def test_lists(self):
        if False:
            while True:
                i = 10
        'Verify the pretty-printing of lists'
        self.assertGdbRepr([])
        self.assertGdbRepr(list(range(5)))

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify the pretty-printing of bytes'
        self.assertGdbRepr(b'')
        self.assertGdbRepr(b'And now for something hopefully the same')
        self.assertGdbRepr(b'string with embedded NUL here \x00 and then some more text')
        self.assertGdbRepr(b'this is a tab:\t this is a slash-N:\n this is a slash-R:\r')
        self.assertGdbRepr(b'this is byte 255:\xff and byte 128:\x80')
        self.assertGdbRepr(bytes([b for b in range(255)]))

    def test_strings(self):
        if False:
            return 10
        'Verify the pretty-printing of unicode strings'
        (out, err) = run_gdb('--eval-command', 'python import locale; print(locale.getpreferredencoding())')
        encoding = out.rstrip()
        if err or not encoding:
            raise RuntimeError(f'unable to determine the preferred encoding of embedded Python in GDB: {err}')

        def check_repr(text):
            if False:
                return 10
            try:
                text.encode(encoding)
            except UnicodeEncodeError:
                self.assertGdbRepr(text, ascii(text))
            else:
                self.assertGdbRepr(text)
        self.assertGdbRepr('')
        self.assertGdbRepr('And now for something hopefully the same')
        self.assertGdbRepr('string with embedded NUL here \x00 and then some more text')
        check_repr('☠')
        check_repr('文字化け')
        check_repr(chr(119073))

    def test_tuples(self):
        if False:
            print('Hello World!')
        'Verify the pretty-printing of tuples'
        self.assertGdbRepr(tuple(), '()')
        self.assertGdbRepr((1,), '(1,)')
        self.assertGdbRepr(('foo', 'bar', 'baz'))

    def test_sets(self):
        if False:
            return 10
        'Verify the pretty-printing of sets'
        if (gdb_major_version, gdb_minor_version) < (7, 3):
            self.skipTest('pretty-printing of sets needs gdb 7.3 or later')
        self.assertGdbRepr(set(), 'set()')
        self.assertGdbRepr(set(['a']), "{'a'}")
        if not sys.flags.ignore_environment:
            self.assertGdbRepr(set(['a', 'b']), "{'a', 'b'}")
            self.assertGdbRepr(set([4, 5, 6]), '{4, 5, 6}')
        (gdb_repr, gdb_output) = self.get_gdb_repr("s = set(['a','b'])\ns.remove('a')\nid(s)")
        self.assertEqual(gdb_repr, "{'b'}")

    def test_frozensets(self):
        if False:
            print('Hello World!')
        'Verify the pretty-printing of frozensets'
        if (gdb_major_version, gdb_minor_version) < (7, 3):
            self.skipTest('pretty-printing of frozensets needs gdb 7.3 or later')
        self.assertGdbRepr(frozenset(), 'frozenset()')
        self.assertGdbRepr(frozenset(['a']), "frozenset({'a'})")
        if not sys.flags.ignore_environment:
            self.assertGdbRepr(frozenset(['a', 'b']), "frozenset({'a', 'b'})")
            self.assertGdbRepr(frozenset([4, 5, 6]), 'frozenset({4, 5, 6})')

    def test_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        (gdb_repr, gdb_output) = self.get_gdb_repr('\ntry:\n    raise RuntimeError("I am an error")\nexcept RuntimeError as e:\n    id(e)\n')
        self.assertEqual(gdb_repr, "RuntimeError('I am an error',)")
        (gdb_repr, gdb_output) = self.get_gdb_repr('\ntry:\n    a = 1 / 0\nexcept ZeroDivisionError as e:\n    id(e)\n')
        self.assertEqual(gdb_repr, "ZeroDivisionError('division by zero',)")

    def test_modern_class(self):
        if False:
            while True:
                i = 10
        'Verify the pretty-printing of new-style class instances'
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo:\n    pass\nfoo = Foo()\nfoo.an_int = 42\nid(foo)')
        m = re.match('<Foo\\(an_int=42\\) at remote 0x-?[0-9a-f]+>', gdb_repr)
        self.assertTrue(m, msg='Unexpected new-style class rendering %r' % gdb_repr)

    def test_subclassing_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify the pretty-printing of an instance of a list subclass'
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo(list):\n    pass\nfoo = Foo()\nfoo += [1, 2, 3]\nfoo.an_int = 42\nid(foo)')
        m = re.match('<Foo\\(an_int=42\\) at remote 0x-?[0-9a-f]+>', gdb_repr)
        self.assertTrue(m, msg='Unexpected new-style class rendering %r' % gdb_repr)

    def test_subclassing_tuple(self):
        if False:
            while True:
                i = 10
        'Verify the pretty-printing of an instance of a tuple subclass'
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo(tuple):\n    pass\nfoo = Foo((1, 2, 3))\nfoo.an_int = 42\nid(foo)')
        m = re.match('<Foo\\(an_int=42\\) at remote 0x-?[0-9a-f]+>', gdb_repr)
        self.assertTrue(m, msg='Unexpected new-style class rendering %r' % gdb_repr)

    def assertSane(self, source, corruption, exprepr=None):
        if False:
            return 10
        "Run Python under gdb, corrupting variables in the inferior process\n        immediately before taking a backtrace.\n\n        Verify that the variable's representation is the expected failsafe\n        representation"
        if corruption:
            cmds_after_breakpoint = [corruption, 'backtrace']
        else:
            cmds_after_breakpoint = ['backtrace']
        (gdb_repr, gdb_output) = self.get_gdb_repr(source, cmds_after_breakpoint=cmds_after_breakpoint)
        if exprepr:
            if gdb_repr == exprepr:
                return
        pattern = '<.* at remote 0x-?[0-9a-f]+>'
        m = re.match(pattern, gdb_repr)
        if not m:
            self.fail('Unexpected gdb representation: %r\n%s' % (gdb_repr, gdb_output))

    def test_NULL_ptr(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that a NULL PyObject* is handled gracefully'
        (gdb_repr, gdb_output) = self.get_gdb_repr('id(42)', cmds_after_breakpoint=['set variable v=0', 'backtrace'])
        self.assertEqual(gdb_repr, '0x0')

    def test_NULL_ob_type(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that a PyObject* with NULL ob_type is handled gracefully'
        self.assertSane('id(42)', 'set v->ob_type=0')

    def test_corrupt_ob_type(self):
        if False:
            return 10
        'Ensure that a PyObject* with a corrupt ob_type is handled gracefully'
        self.assertSane('id(42)', 'set v->ob_type=0xDEADBEEF', exprepr='42')

    def test_corrupt_tp_flags(self):
        if False:
            return 10
        'Ensure that a PyObject* with a type with corrupt tp_flags is handled'
        self.assertSane('id(42)', 'set v->ob_type->tp_flags=0x0', exprepr='42')

    def test_corrupt_tp_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that a PyObject* with a type with corrupt tp_name is handled'
        self.assertSane('id(42)', 'set v->ob_type->tp_name=0xDEADBEEF', exprepr='42')

    def test_builtins_help(self):
        if False:
            i = 10
            return i + 15
        'Ensure that the new-style class _Helper in site.py can be handled'
        if sys.flags.no_site:
            self.skipTest('need site module, but -S option was used')
        (gdb_repr, gdb_output) = self.get_gdb_repr('id(__builtins__.help)', import_site=True)
        m = re.match('<_Helper at remote 0x-?[0-9a-f]+>', gdb_repr)
        self.assertTrue(m, msg='Unexpected rendering %r' % gdb_repr)

    def test_selfreferential_list(self):
        if False:
            print('Hello World!')
        "Ensure that a reference loop involving a list doesn't lead proxyval\n        into an infinite loop:"
        (gdb_repr, gdb_output) = self.get_gdb_repr('a = [3, 4, 5] ; a.append(a) ; id(a)')
        self.assertEqual(gdb_repr, '[3, 4, 5, [...]]')
        (gdb_repr, gdb_output) = self.get_gdb_repr('a = [3, 4, 5] ; b = [a] ; a.append(b) ; id(a)')
        self.assertEqual(gdb_repr, '[3, 4, 5, [[...]]]')

    def test_selfreferential_dict(self):
        if False:
            i = 10
            return i + 15
        "Ensure that a reference loop involving a dict doesn't lead proxyval\n        into an infinite loop:"
        (gdb_repr, gdb_output) = self.get_gdb_repr("a = {} ; b = {'bar':a} ; a['foo'] = b ; id(a)")
        self.assertEqual(gdb_repr, "{'foo': {'bar': {...}}}")

    def test_selfreferential_old_style_instance(self):
        if False:
            i = 10
            return i + 15
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo:\n    pass\nfoo = Foo()\nfoo.an_attr = foo\nid(foo)')
        self.assertTrue(re.match('<Foo\\(an_attr=<\\.\\.\\.>\\) at remote 0x-?[0-9a-f]+>', gdb_repr), 'Unexpected gdb representation: %r\n%s' % (gdb_repr, gdb_output))

    def test_selfreferential_new_style_instance(self):
        if False:
            i = 10
            return i + 15
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo(object):\n    pass\nfoo = Foo()\nfoo.an_attr = foo\nid(foo)')
        self.assertTrue(re.match('<Foo\\(an_attr=<\\.\\.\\.>\\) at remote 0x-?[0-9a-f]+>', gdb_repr), 'Unexpected gdb representation: %r\n%s' % (gdb_repr, gdb_output))
        (gdb_repr, gdb_output) = self.get_gdb_repr('\nclass Foo(object):\n    pass\na = Foo()\nb = Foo()\na.an_attr = b\nb.an_attr = a\nid(a)')
        self.assertTrue(re.match('<Foo\\(an_attr=<Foo\\(an_attr=<\\.\\.\\.>\\) at remote 0x-?[0-9a-f]+>\\) at remote 0x-?[0-9a-f]+>', gdb_repr), 'Unexpected gdb representation: %r\n%s' % (gdb_repr, gdb_output))

    def test_truncation(self):
        if False:
            print('Hello World!')
        'Verify that very long output is truncated'
        (gdb_repr, gdb_output) = self.get_gdb_repr('id(list(range(1000)))')
        self.assertEqual(gdb_repr, '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226...(truncated)')
        self.assertEqual(len(gdb_repr), 1024 + len('...(truncated)'))

    def test_builtin_method(self):
        if False:
            for i in range(10):
                print('nop')
        (gdb_repr, gdb_output) = self.get_gdb_repr('import sys; id(sys.stdout.readlines)')
        self.assertTrue(re.match('<built-in method readlines of _io.TextIOWrapper object at remote 0x-?[0-9a-f]+>', gdb_repr), 'Unexpected gdb representation: %r\n%s' % (gdb_repr, gdb_output))

    def test_frames(self):
        if False:
            for i in range(10):
                print('nop')
        gdb_output = self.get_stack_trace('\ndef foo(a, b, c):\n    pass\n\nfoo(3, 4, 5)\nid(foo.__code__)', breakpoint='builtin_id', cmds_after_breakpoint=['print (PyFrameObject*)(((PyCodeObject*)v)->co_mutable->co_zombieframe)'])
        self.assertTrue(re.match('.*\\s+\\$1 =\\s+Frame 0x-?[0-9a-f]+, for file <string>, line 3, in foo \\(\\)\\s+.*', gdb_output, re.DOTALL), 'Unexpected gdb representation: %r\n%s' % (gdb_output, gdb_output))

@unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
class PyListTests(DebuggerTests):

    def assertListing(self, expected, actual):
        if False:
            i = 10
            return i + 15
        self.assertEndsWith(actual, expected)

    def test_basic_command(self):
        if False:
            return 10
        'Verify that the "py-list" command works'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-list'])
        self.assertListing('   5    \n   6    def bar(a, b, c):\n   7        baz(a, b, c)\n   8    \n   9    def baz(*args):\n >10        id(42)\n  11    \n  12    foo(1, 2, 3)\n', bt)

    def test_one_abs_arg(self):
        if False:
            print('Hello World!')
        'Verify the "py-list" command with one absolute argument'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-list 9'])
        self.assertListing('   9    def baz(*args):\n >10        id(42)\n  11    \n  12    foo(1, 2, 3)\n', bt)

    def test_two_abs_args(self):
        if False:
            i = 10
            return i + 15
        'Verify the "py-list" command with two absolute arguments'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-list 1,3'])
        self.assertListing('   1    # Sample script for use by test_gdb.py\n   2    \n   3    def foo(a, b, c):\n', bt)

class StackNavigationTests(DebuggerTests):

    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_pyup_command(self):
        if False:
            while True:
                i = 10
        'Verify that the "py-up" command works'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-up'])
        self.assertMultilineMatches(bt, '^.*\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 7, in bar \\(a=1, b=2, c=3\\)\n    baz\\(a, b, c\\)\n$')

    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    def test_down_at_bottom(self):
        if False:
            while True:
                i = 10
        'Verify handling of "py-down" at the bottom of the stack'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-down'])
        self.assertEndsWith(bt, 'Unable to find a newer python frame\n')

    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    def test_up_at_top(self):
        if False:
            return 10
        'Verify handling of "py-up" at the top of the stack'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up'] * 5)
        self.assertEndsWith(bt, 'Unable to find an older python frame\n')

    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_up_then_down(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify "py-up" followed by "py-down"'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-up', 'py-down'])
        self.assertMultilineMatches(bt, '^.*\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 7, in bar \\(a=1, b=2, c=3\\)\n    baz\\(a, b, c\\)\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 10, in baz \\(args=\\(1, 2, 3\\)\\)\n    id\\(42\\)\n$')

class PyBtTests(DebuggerTests):

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_bt(self):
        if False:
            i = 10
            return i + 15
        'Verify that the "py-bt" command works'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-bt'])
        self.assertMultilineMatches(bt, '^.*\nTraceback \\(most recent call first\\):\n  <built-in method id of module object .*>\n  File ".*gdb_sample.py", line 10, in baz\n    id\\(42\\)\n  File ".*gdb_sample.py", line 7, in bar\n    baz\\(a, b, c\\)\n  File ".*gdb_sample.py", line 4, in foo\n    bar\\(a, b, c\\)\n  File ".*gdb_sample.py", line 12, in <module>\n    foo\\(1, 2, 3\\)\n')

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_bt_full(self):
        if False:
            while True:
                i = 10
        'Verify that the "py-bt-full" command works'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-bt-full'])
        self.assertMultilineMatches(bt, '^.*\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 7, in bar \\(a=1, b=2, c=3\\)\n    baz\\(a, b, c\\)\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 4, in foo \\(a=1, b=2, c=3\\)\n    bar\\(a, b, c\\)\n#[0-9]+ Frame 0x-?[0-9a-f]+, for file .*gdb_sample.py, line 12, in <module> \\(\\)\n    foo\\(1, 2, 3\\)\n')

    def test_threads(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that "py-bt" indicates threads that are waiting for the GIL'
        cmd = "\nfrom threading import Thread\n\nclass TestThread(Thread):\n    # These threads would run forever, but we'll interrupt things with the\n    # debugger\n    def run(self):\n        i = 0\n        while 1:\n             i += 1\n\nt = {}\nfor i in range(4):\n   t[i] = TestThread()\n   t[i].start()\n\n# Trigger a breakpoint on the main thread\nid(42)\n\n"
        gdb_output = self.get_stack_trace(cmd, cmds_after_breakpoint=['thread apply all py-bt'])
        self.assertIn('Waiting for the GIL', gdb_output)
        gdb_output = self.get_stack_trace(cmd, cmds_after_breakpoint=['thread apply all py-bt-full'])
        self.assertIn('Waiting for the GIL', gdb_output)

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_gc(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that "py-bt" indicates if a thread is garbage-collecting'
        cmd = 'from gc import collect\nid(42)\ndef foo():\n    collect()\ndef bar():\n    foo()\nbar()\n'
        gdb_output = self.get_stack_trace(cmd, cmds_after_breakpoint=['break update_refs', 'continue', 'py-bt'])
        self.assertIn('Garbage-collecting', gdb_output)
        gdb_output = self.get_stack_trace(cmd, cmds_after_breakpoint=['break update_refs', 'continue', 'py-bt-full'])
        self.assertIn('Garbage-collecting', gdb_output)

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_pycfunction(self):
        if False:
            i = 10
            return i + 15
        'Verify that "py-bt" displays invocations of PyCFunction instances'
        for (func_name, args, expected_frame) in (('meth_varargs', '', 1), ('meth_varargs_keywords', '', 1), ('meth_o', '[]', 1), ('meth_noargs', '', 1), ('meth_fastcall', '', 1), ('meth_fastcall_keywords', '', 1)):
            for obj in ('_testcapi', '_testcapi.MethClass', '_testcapi.MethClass()', '_testcapi.MethStatic()'):
                with self.subTest(f'{obj}.{func_name}'):
                    cmd = textwrap.dedent(f'\n                        import _testcapi\n                        def foo():\n                            {obj}.{func_name}({args})\n                        def bar():\n                            foo()\n                        bar()\n                    ')
                    gdb_output = self.get_stack_trace(cmd, breakpoint=func_name, cmds_after_breakpoint=['bt', 'py-bt'], ignore_stderr=True)
                    self.assertIn(f'<built-in method {func_name}', gdb_output)
                    gdb_output = self.get_stack_trace(cmd, breakpoint=func_name, cmds_after_breakpoint=['py-bt-full'], ignore_stderr=True)
                    self.assertIn(f'#{expected_frame} <built-in method {func_name}', gdb_output)

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_wrapper_call(self):
        if False:
            return 10
        cmd = textwrap.dedent('\n            class MyList(list):\n                def __init__(self):\n                    super().__init__()   # wrapper_call()\n\n            id("first break point")\n            l = MyList()\n        ')
        cmds_after_breakpoint = ['break wrapper_call', 'continue']
        if CET_PROTECTION:
            cmds_after_breakpoint.append('next')
        cmds_after_breakpoint.append('py-bt')
        gdb_output = self.get_stack_trace(cmd, cmds_after_breakpoint=cmds_after_breakpoint)
        self.assertRegex(gdb_output, "<method-wrapper u?'__init__' of MyList object at ")

class PyPrintTests(DebuggerTests):

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_basic_command(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that the "py-print" command works'
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-print args'])
        self.assertMultilineMatches(bt, ".*\\nlocal 'args' = \\(1, 2, 3\\)\\n.*")

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    def test_print_after_up(self):
        if False:
            return 10
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-up', 'py-print c', 'py-print b', 'py-print a'])
        self.assertMultilineMatches(bt, ".*\\nlocal 'c' = 3\\nlocal 'b' = 2\\nlocal 'a' = 1\\n.*")

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_printing_global(self):
        if False:
            return 10
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-print __name__'])
        self.assertMultilineMatches(bt, ".*\\nglobal '__name__' = '__main__'\\n.*")

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_printing_builtin(self):
        if False:
            while True:
                i = 10
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-print len'])
        self.assertMultilineMatches(bt, ".*\\nbuiltin 'len' = <built-in method len of module object at remote 0x-?[0-9a-f]+>\\n.*")

class PyLocalsTests(DebuggerTests):

    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_basic_command(self):
        if False:
            i = 10
            return i + 15
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-locals'])
        self.assertMultilineMatches(bt, '.*\\nargs = \\(1, 2, 3\\)\\n.*')

    @unittest.skipUnless(HAS_PYUP_PYDOWN, 'test requires py-up/py-down commands')
    @unittest.skipIf(python_is_optimized(), 'Python was compiled with optimizations')
    def test_locals_after_up(self):
        if False:
            return 10
        bt = self.get_stack_trace(script=self.get_sample_script(), cmds_after_breakpoint=['py-up', 'py-up', 'py-locals'])
        self.assertMultilineMatches(bt, '.*\\na = 1\\nb = 2\\nc = 3\\n.*')

def setUpModule():
    if False:
        while True:
            i = 10
    if support.verbose:
        print('GDB version %s.%s:' % (gdb_major_version, gdb_minor_version))
        for line in gdb_version.splitlines():
            print(' ' * 4 + line)
if __name__ == '__main__':
    unittest.main()