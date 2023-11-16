import asyncio
import errno
import fcntl
import inspect
import logging
import os
import platform
import pty
import resource
import signal
import stat
import sys
import tempfile
import time
import unittest
import unittest.mock
import warnings
from asyncio.queues import Queue as AQueue
from contextlib import contextmanager
from functools import partial, wraps
from hashlib import md5
from io import BytesIO, StringIO
from os.path import dirname, exists, join, realpath, split
from pathlib import Path
import sh
THIS_DIR = Path(__file__).resolve().parent
RAND_BYTES = os.urandom(10)
tempdir = Path(tempfile.gettempdir()).resolve()
IS_MACOS = platform.system() in ('AIX', 'Darwin')

def hash(a: str):
    if False:
        for i in range(10):
            print('nop')
    h = md5(a.encode('utf8') + RAND_BYTES)
    return h.hexdigest()

def randomize_order(a, b):
    if False:
        while True:
            i = 10
    h1 = hash(a)
    h2 = hash(b)
    if h1 == h2:
        return 0
    elif h1 < h2:
        return -1
    else:
        return 1
unittest.TestLoader.sortTestMethodsUsing = staticmethod(randomize_order)

def append_pythonpath(env, path):
    if False:
        print('Hello World!')
    key = 'PYTHONPATH'
    pypath = [p for p in env.get(key, '').split(':') if p]
    pypath.insert(0, path)
    pypath = ':'.join(pypath)
    env[key] = pypath

def get_module_import_dir(m):
    if False:
        while True:
            i = 10
    mod_file = inspect.getsourcefile(m)
    is_package = mod_file.endswith('__init__.py')
    mod_dir = dirname(mod_file)
    if is_package:
        (mod_dir, _) = split(mod_dir)
    return mod_dir

def append_module_path(env, m):
    if False:
        return 10
    append_pythonpath(env, get_module_import_dir(m))
system_python = sh.Command(sys.executable)
baked_env = os.environ.copy()
append_module_path(baked_env, sh)
python = system_python.bake(_env=baked_env, _return_cmd=True)
pythons = python.bake(_return_cmd=False)

def requires_progs(*progs):
    if False:
        print('Hello World!')
    missing = []
    for prog in progs:
        try:
            sh.Command(prog)
        except sh.CommandNotFound:
            missing.append(prog)
    friendly_missing = ', '.join(missing)
    return unittest.skipUnless(len(missing) == 0, f'Missing required system programs: {friendly_missing}')
requires_posix = unittest.skipUnless(os.name == 'posix', 'Requires POSIX')
requires_utf8 = unittest.skipUnless(sh.DEFAULT_ENCODING == 'UTF-8', 'System encoding must be UTF-8')
not_macos = unittest.skipUnless(not IS_MACOS, "Doesn't work on MacOS")

def requires_poller(poller):
    if False:
        return 10
    use_select = bool(int(os.environ.get('SH_TESTS_USE_SELECT', '0')))
    cur_poller = 'select' if use_select else 'poll'
    return unittest.skipUnless(cur_poller == poller, f'Only enabled for select.{cur_poller}')

@contextmanager
def ulimit(key, new_soft):
    if False:
        print('Hello World!')
    (soft, hard) = resource.getrlimit(key)
    resource.setrlimit(key, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(key, (soft, hard))

def create_tmp_test(code, prefix='tmp', delete=True, **kwargs):
    if False:
        print('Hello World!')
    'creates a temporary test file that lives on disk, on which we can run\n    python with sh'
    py = tempfile.NamedTemporaryFile(prefix=prefix, delete=delete)
    code = code.format(**kwargs)
    code = code.encode('UTF-8')
    py.write(code)
    py.flush()
    st = os.stat(py.name)
    os.chmod(py.name, st.st_mode | stat.S_IEXEC)
    return py

class BaseTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.simplefilter('ignore', ResourceWarning)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        warnings.simplefilter('default', ResourceWarning)

    def assert_oserror(self, num, fn, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            fn(*args, **kwargs)
        except OSError as e:
            self.assertEqual(e.errno, num)

    def assert_deprecated(self, fn, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            fn(*args, **kwargs)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

class ArgTests(BaseTests):

    def test_list_args(self):
        if False:
            i = 10
            return i + 15
        processed = sh._aggregate_keywords({'arg': [1, 2, 3]}, '=', '--')
        self.assertListEqual(processed, ['--arg=1', '--arg=2', '--arg=3'])

    def test_bool_values(self):
        if False:
            for i in range(10):
                print('nop')
        processed = sh._aggregate_keywords({'truthy': True, 'falsey': False}, '=', '--')
        self.assertListEqual(processed, ['--truthy'])

    def test_space_sep(self):
        if False:
            return 10
        processed = sh._aggregate_keywords({'arg': '123'}, ' ', '--')
        self.assertListEqual(processed, ['--arg', '123'])

@requires_posix
class FunctionalTests(BaseTests):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._environ = os.environ.copy()
        super().setUp()

    def tearDown(self):
        if False:
            print('Hello World!')
        os.environ = self._environ
        super().tearDown()

    def test_print_command(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import ls, which
        actual_location = which('ls').strip()
        out = str(ls)
        self.assertEqual(out, actual_location)

    def test_unicode_arg(self):
        if False:
            print('Hello World!')
        from sh import echo
        test = '漢字'
        p = echo(test, _encoding='utf8')
        output = p.strip()
        self.assertEqual(test, output)

    def test_unicode_exception(self):
        if False:
            i = 10
            return i + 15
        from sh import ErrorReturnCode
        py = create_tmp_test('exit(1)')
        arg = '漢字'
        native_arg = arg
        try:
            python(py.name, arg, _encoding='utf8')
        except ErrorReturnCode as e:
            self.assertIn(native_arg, str(e))
        else:
            self.fail("exception wasn't raised")

    def test_pipe_fd(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('print("hi world")')
        (read_fd, write_fd) = os.pipe()
        python(py.name, _out=write_fd)
        out = os.read(read_fd, 10)
        self.assertEqual(out, b'hi world\n')

    def test_trunc_exc(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport sys\nsys.stdout.write("a" * 1000)\nsys.stderr.write("b" * 1000)\nexit(1)\n')
        self.assertRaises(sh.ErrorReturnCode_1, python, py.name)

    def test_number_arg(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\noptions, args = parser.parse_args()\nprint(args[0])\n')
        out = python(py.name, 3).strip()
        self.assertEqual(out, '3')

    def test_arg_string_coercion(self):
        if False:
            return 10
        py = create_tmp_test('\nfrom argparse import ArgumentParser\nparser = ArgumentParser()\nparser.add_argument("-n", type=int)\nparser.add_argument("--number", type=int)\nns = parser.parse_args()\nprint(ns.n + ns.number)\n')
        out = python(py.name, n=3, number=4, _long_sep=None).strip()
        self.assertEqual(out, '7')

    def test_empty_stdin_no_hang(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\ndata = sys.stdin.read()\nsys.stdout.write("no hang")\n')
        out = pythons(py.name, _in='', _timeout=2)
        self.assertEqual(out, 'no hang')
        out = pythons(py.name, _in=None, _timeout=2)
        self.assertEqual(out, 'no hang')

    def test_exit_code(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import ErrorReturnCode_3
        py = create_tmp_test('\nexit(3)\n')
        self.assertRaises(ErrorReturnCode_3, python, py.name)

    def test_patched_glob(self):
        if False:
            for i in range(10):
                print('nop')
        from glob import glob
        py = create_tmp_test('\nimport sys\nprint(sys.argv[1:])\n')
        files = glob('*.faowjefoajweofj')
        out = python(py.name, files).strip()
        self.assertEqual(out, "['*.faowjefoajweofj']")

    def test_exit_code_with_hasattr(self):
        if False:
            return 10
        from sh import ErrorReturnCode_3
        py = create_tmp_test('\nexit(3)\n')
        try:
            out = python(py.name, _iter=True)
            hasattr(out, 'something_not_there')
            list(out)
            self.assertEqual(out.exit_code, 3)
            self.fail('Command exited with error, but no exception thrown')
        except ErrorReturnCode_3:
            pass

    def test_exit_code_from_exception(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import ErrorReturnCode_3
        py = create_tmp_test('\nexit(3)\n')
        self.assertRaises(ErrorReturnCode_3, python, py.name)
        try:
            python(py.name)
        except Exception as e:
            self.assertEqual(e.exit_code, 3)

    def test_stdin_from_string(self):
        if False:
            while True:
                i = 10
        from sh import sed
        self.assertEqual(sed(_in='one test three', e='s/test/two/').strip(), 'one two three')

    def test_ok_code(self):
        if False:
            return 10
        from sh import ErrorReturnCode_1, ErrorReturnCode_2, ls
        exc_to_test = ErrorReturnCode_2
        code_to_pass = 2
        if IS_MACOS:
            exc_to_test = ErrorReturnCode_1
            code_to_pass = 1
        self.assertRaises(exc_to_test, ls, '/aofwje/garogjao4a/eoan3on')
        ls('/aofwje/garogjao4a/eoan3on', _ok_code=code_to_pass)
        ls('/aofwje/garogjao4a/eoan3on', _ok_code=[code_to_pass])
        ls('/aofwje/garogjao4a/eoan3on', _ok_code=range(code_to_pass + 1))

    def test_ok_code_none(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('exit(0)')
        python(py.name, _ok_code=None)

    def test_ok_code_exception(self):
        if False:
            i = 10
            return i + 15
        from sh import ErrorReturnCode_0
        py = create_tmp_test('exit(0)')
        self.assertRaises(ErrorReturnCode_0, python, py.name, _ok_code=2)

    def test_none_arg(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nprint(sys.argv[1:])\n')
        maybe_arg = 'some'
        out = python(py.name, maybe_arg).strip()
        self.assertEqual(out, "['some']")
        maybe_arg = None
        out = python(py.name, maybe_arg).strip()
        self.assertEqual(out, '[]')

    def test_quote_escaping(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\noptions, args = parser.parse_args()\nprint(args)\n')
        out = python(py.name, 'one two three').strip()
        self.assertEqual(out, "['one two three']")
        out = python(py.name, 'one "two three').strip()
        self.assertEqual(out, '[\'one "two three\']')
        out = python(py.name, 'one', 'two three').strip()
        self.assertEqual(out, "['one', 'two three']")
        out = python(py.name, 'one', 'two "haha" three').strip()
        self.assertEqual(out, '[\'one\', \'two "haha" three\']')
        out = python(py.name, "one two's three").strip()
        self.assertEqual(out, '["one two\'s three"]')
        out = python(py.name, "one two's three").strip()
        self.assertEqual(out, '["one two\'s three"]')

    def test_multiple_pipes(self):
        if False:
            return 10
        import time
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor l in "andrew":\n    sys.stdout.write(l)\n    time.sleep(.2)\n')
        inc_py = create_tmp_test('\nimport sys\nwhile True:\n    letter = sys.stdin.read(1)\n    if not letter:\n        break\n    sys.stdout.write(chr(ord(letter)+1))\n')

        def inc(*args, **kwargs):
            if False:
                return 10
            return python('-u', inc_py.name, *args, **kwargs)

        class Derp(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.times = []
                self.stdout = []
                self.last_received = None

            def agg(self, line):
                if False:
                    return 10
                self.stdout.append(line.strip())
                now = time.time()
                if self.last_received:
                    self.times.append(now - self.last_received)
                self.last_received = now
        derp = Derp()
        p = inc(_in=inc(_in=inc(_in=python('-u', py.name, _piped=True), _piped=True), _piped=True), _out=derp.agg)
        p.wait()
        self.assertEqual(''.join(derp.stdout), 'dqguhz')
        self.assertTrue(all([t > 0.15 for t in derp.times]))

    def test_manual_stdin_string(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import tr
        out = tr('[:lower:]', '[:upper:]', _in='andrew').strip()
        self.assertEqual(out, 'ANDREW')

    def test_manual_stdin_iterable(self):
        if False:
            i = 10
            return i + 15
        from sh import tr
        test = ['testing\n', 'herp\n', 'derp\n']
        out = tr('[:lower:]', '[:upper:]', _in=test)
        match = ''.join([t.upper() for t in test])
        self.assertEqual(out, match)

    def test_manual_stdin_file(self):
        if False:
            return 10
        import tempfile
        from sh import tr
        test_string = 'testing\nherp\nderp\n'
        stdin = tempfile.NamedTemporaryFile()
        stdin.write(test_string.encode())
        stdin.flush()
        stdin.seek(0)
        out = tr('[:lower:]', '[:upper:]', _in=stdin)
        self.assertEqual(out, test_string.upper())

    def test_manual_stdin_queue(self):
        if False:
            print('Hello World!')
        from sh import tr
        try:
            from Queue import Queue
        except ImportError:
            from queue import Queue
        test = ['testing\n', 'herp\n', 'derp\n']
        q = Queue()
        for t in test:
            q.put(t)
        q.put(None)
        out = tr('[:lower:]', '[:upper:]', _in=q)
        match = ''.join([t.upper() for t in test])
        self.assertEqual(out, match)

    def test_environment(self):
        if False:
            i = 10
            return i + 15
        'tests that environments variables that we pass into sh commands\n        exist in the environment, and on the sh module'
        import os
        env = {'HERP': 'DERP'}
        py = create_tmp_test('\nimport os\n\nfor key in list(os.environ.keys()):\n    if key != "HERP":\n        del os.environ[key]\nprint(dict(os.environ))\n')
        out = python(py.name, _env=env).strip()
        self.assertEqual(out, "{'HERP': 'DERP'}")
        py = create_tmp_test('\nimport os, sys\nsys.path.insert(0, os.getcwd())\nimport sh\nfor key in list(os.environ.keys()):\n    if key != "HERP":\n        del os.environ[key]\nprint(dict(HERP=sh.HERP))\n')
        out = python(py.name, _env=env, _cwd=THIS_DIR).strip()
        self.assertEqual(out, "{'HERP': 'DERP'}")
        os.environ['HERP'] = 'DERP'
        out = python(py.name, _env=os.environ, _cwd=THIS_DIR).strip()
        self.assertEqual(out, "{'HERP': 'DERP'}")

    def test_which(self):
        if False:
            i = 10
            return i + 15
        from sh import ls
        which = sh._SelfWrapper__env.b_which
        self.assertEqual(which('fjoawjefojawe'), None)
        self.assertEqual(which('ls'), str(ls))

    def test_which_paths(self):
        if False:
            print('Hello World!')
        which = sh._SelfWrapper__env.b_which
        py = create_tmp_test('\nprint("hi")\n')
        test_path = dirname(py.name)
        (_, test_name) = os.path.split(py.name)
        found_path = which(test_name)
        self.assertEqual(found_path, None)
        found_path = which(test_name, [test_path])
        self.assertEqual(found_path, py.name)

    def test_no_close_fds(self):
        if False:
            for i in range(10):
                print('nop')
        tmp = [tempfile.TemporaryFile() for i in range(10)]
        for t in tmp:
            flags = fcntl.fcntl(t.fileno(), fcntl.F_GETFD)
            flags &= ~fcntl.FD_CLOEXEC
            fcntl.fcntl(t.fileno(), fcntl.F_SETFD, flags)
        py = create_tmp_test('\nimport os\nprint(len(os.listdir("/dev/fd")))\n')
        out = python(py.name, _close_fds=False).strip()
        self.assertGreater(int(out), 7)
        for t in tmp:
            t.close()

    def test_close_fds(self):
        if False:
            while True:
                i = 10
        tmp = [tempfile.TemporaryFile() for i in range(10)]
        for t in tmp:
            flags = fcntl.fcntl(t.fileno(), fcntl.F_GETFD)
            flags &= ~fcntl.FD_CLOEXEC
            fcntl.fcntl(t.fileno(), fcntl.F_SETFD, flags)
        py = create_tmp_test('\nimport os\nprint(os.listdir("/dev/fd"))\n')
        out = python(py.name).strip()
        self.assertEqual(out, "['0', '1', '2', '3']")
        for t in tmp:
            t.close()

    def test_pass_fds(self):
        if False:
            print('Hello World!')
        tmp = [tempfile.TemporaryFile() for i in range(10)]
        for t in tmp:
            flags = fcntl.fcntl(t.fileno(), fcntl.F_GETFD)
            flags &= ~fcntl.FD_CLOEXEC
            fcntl.fcntl(t.fileno(), fcntl.F_SETFD, flags)
        last_fd = tmp[-1].fileno()
        py = create_tmp_test('\nimport os\nprint(os.listdir("/dev/fd"))\n')
        out = python(py.name, _pass_fds=[last_fd]).strip()
        inherited = [0, 1, 2, 3, last_fd]
        inherited_str = [str(i) for i in inherited]
        self.assertEqual(out, str(inherited_str))
        for t in tmp:
            t.close()

    def test_no_arg(self):
        if False:
            for i in range(10):
                print('nop')
        import pwd
        from sh import whoami
        u1 = whoami().strip()
        u2 = pwd.getpwuid(os.geteuid())[0]
        self.assertEqual(u1, u2)

    def test_incompatible_special_args(self):
        if False:
            print('Hello World!')
        from sh import ls
        self.assertRaises(TypeError, ls, _iter=True, _piped=True)

    def test_invalid_env(self):
        if False:
            i = 10
            return i + 15
        from sh import ls
        exc = TypeError
        self.assertRaises(exc, ls, _env='XXX')
        self.assertRaises(exc, ls, _env={'foo': 123})
        self.assertRaises(exc, ls, _env={123: 'bar'})

    def test_exception(self):
        if False:
            while True:
                i = 10
        from sh import ErrorReturnCode_2
        py = create_tmp_test('\nexit(2)\n')
        self.assertRaises(ErrorReturnCode_2, python, py.name)

    def test_piped_exception1(self):
        if False:
            return 10
        from sh import ErrorReturnCode_2
        py = create_tmp_test('\nimport sys\nsys.stdout.write("line1\\n")\nsys.stdout.write("line2\\n")\nsys.stdout.flush()\nexit(2)\n')
        py2 = create_tmp_test('')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            list(python(python(py.name, _piped=True), '-u', py2.name, _iter=True))
        self.assertRaises(ErrorReturnCode_2, fn)

    def test_piped_exception2(self):
        if False:
            i = 10
            return i + 15
        from sh import ErrorReturnCode_2
        py = create_tmp_test('\nimport sys\nsys.stdout.write("line1\\n")\nsys.stdout.write("line2\\n")\nsys.stdout.flush()\nexit(2)\n')
        py2 = create_tmp_test('')

        def fn():
            if False:
                i = 10
                return i + 15
            python(python(py.name, _piped=True), '-u', py2.name)
        self.assertRaises(ErrorReturnCode_2, fn)

    def test_command_not_found(self):
        if False:
            i = 10
            return i + 15
        from sh import CommandNotFound

        def do_import():
            if False:
                for i in range(10):
                    print('nop')
            from sh import aowjgoawjoeijaowjellll
        self.assertRaises(ImportError, do_import)

        def do_import():
            if False:
                for i in range(10):
                    print('nop')
            import sh
            sh.awoefaowejfw
        self.assertRaises(CommandNotFound, do_import)

        def do_import():
            if False:
                print('Hello World!')
            import sh
            sh.Command('ofajweofjawoe')
        self.assertRaises(CommandNotFound, do_import)

    def test_command_wrapper_equivalence(self):
        if False:
            i = 10
            return i + 15
        from sh import Command, ls, which
        self.assertEqual(Command(str(which('ls')).strip()), ls)

    def test_doesnt_execute_directories(self):
        if False:
            while True:
                i = 10
        save_path = os.environ['PATH']
        bin_dir1 = tempfile.mkdtemp()
        bin_dir2 = tempfile.mkdtemp()
        gcc_dir1 = os.path.join(bin_dir1, 'gcc')
        gcc_file2 = os.path.join(bin_dir2, 'gcc')
        try:
            os.environ['PATH'] = os.pathsep.join((bin_dir1, bin_dir2))
            os.makedirs(gcc_dir1)
            bunk_header = '#!/bin/sh\necho $*'
            with open(gcc_file2, 'w') as h:
                h.write(bunk_header)
            os.chmod(gcc_file2, int(493))
            from sh import gcc
            self.assertEqual(gcc._path, gcc_file2)
            self.assertEqual(gcc('no-error', _return_cmd=True).stdout.strip(), 'no-error'.encode('ascii'))
        finally:
            os.environ['PATH'] = save_path
            if exists(gcc_file2):
                os.unlink(gcc_file2)
            if exists(gcc_dir1):
                os.rmdir(gcc_dir1)
            if exists(bin_dir1):
                os.rmdir(bin_dir1)
            if exists(bin_dir1):
                os.rmdir(bin_dir2)

    def test_multiple_args_short_option(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("-l", dest="long_option")\noptions, args = parser.parse_args()\nprint(len(options.long_option.split()))\n')
        num_args = int(python(py.name, l='one two three'))
        self.assertEqual(num_args, 3)
        num_args = int(python(py.name, '-l', "one's two's three's"))
        self.assertEqual(num_args, 3)

    def test_multiple_args_long_option(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("-l", "--long-option", dest="long_option")\noptions, args = parser.parse_args()\nprint(len(options.long_option.split()))\n')
        num_args = int(python(py.name, long_option='one two three', nothing=False))
        self.assertEqual(num_args, 3)
        num_args = int(python(py.name, '--long-option', "one's two's three's"))
        self.assertEqual(num_args, 3)

    def test_short_bool_option(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("-s", action="store_true", default=False, dest="short_option")\noptions, args = parser.parse_args()\nprint(options.short_option)\n')
        self.assertTrue(python(py.name, s=True).strip() == 'True')
        self.assertTrue(python(py.name, s=False).strip() == 'False')
        self.assertTrue(python(py.name).strip() == 'False')

    def test_long_bool_option(self):
        if False:
            return 10
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("-l", "--long-option", action="store_true", default=False,     dest="long_option")\noptions, args = parser.parse_args()\nprint(options.long_option)\n')
        self.assertTrue(python(py.name, long_option=True).strip() == 'True')
        self.assertTrue(python(py.name).strip() == 'False')

    def test_false_bool_ignore(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nprint(sys.argv[1:])\n')
        test = True
        self.assertEqual(python(py.name, test and '-n').strip(), "['-n']")
        test = False
        self.assertEqual(python(py.name, test and '-n').strip(), '[]')

    def test_composition(self):
        if False:
            while True:
                i = 10
        py1 = create_tmp_test('\nimport sys\nprint(int(sys.argv[1]) * 2)\n        ')
        py2 = create_tmp_test('\nimport sys\nprint(int(sys.argv[1]) + 1)\n        ')
        res = python(py2.name, python(py1.name, 8)).strip()
        self.assertEqual('17', res)

    def test_incremental_composition(self):
        if False:
            while True:
                i = 10
        py1 = create_tmp_test('\nimport sys\nprint(int(sys.argv[1]) * 2)\n        ')
        py2 = create_tmp_test('\nimport sys\nprint(int(sys.stdin.read()) + 1)\n        ')
        res = python(py2.name, _in=python(py1.name, 8, _piped=True)).strip()
        self.assertEqual('17', res)

    def test_short_option(self):
        if False:
            i = 10
            return i + 15
        from sh import sh
        s1 = sh(c='echo test').strip()
        s2 = 'test'
        self.assertEqual(s1, s2)

    def test_long_option(self):
        if False:
            return 10
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("-l", "--long-option", action="store", default="", dest="long_option")\noptions, args = parser.parse_args()\nprint(options.long_option.upper())\n')
        self.assertTrue(python(py.name, long_option='testing').strip() == 'TESTING')
        self.assertTrue(python(py.name).strip() == '')

    def test_raw_args(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nfrom optparse import OptionParser\nparser = OptionParser()\nparser.add_option("--long_option", action="store", default=None,\n    dest="long_option1")\nparser.add_option("--long-option", action="store", default=None,\n    dest="long_option2")\noptions, args = parser.parse_args()\n\nif options.long_option1:\n    print(options.long_option1.upper())\nelse:\n    print(options.long_option2.upper())\n')
        self.assertEqual(python(py.name, {'long_option': 'underscore'}).strip(), 'UNDERSCORE')
        self.assertEqual(python(py.name, long_option='hyphen').strip(), 'HYPHEN')

    def test_custom_separator(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nprint(sys.argv[1])\n')
        opt = {'long-option': 'underscore'}
        correct = '--long-option=custom=underscore'
        out = python(py.name, opt, _long_sep='=custom=').strip()
        self.assertEqual(out, correct)
        correct = '--long-option=baked=underscore'
        python_baked = python.bake(py.name, opt, _long_sep='=baked=')
        out = python_baked().strip()
        self.assertEqual(out, correct)

    def test_custom_separator_space(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nprint(str(sys.argv[1:]))\n')
        opt = {'long-option': 'space'}
        correct = ['--long-option', 'space']
        out = python(py.name, opt, _long_sep=' ').strip()
        self.assertEqual(out, str(correct))

    def test_custom_long_prefix(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nprint(sys.argv[1])\n')
        out = python(py.name, {'long-option': 'underscore'}, _long_prefix='-custom-').strip()
        self.assertEqual(out, '-custom-long-option=underscore')
        out = python(py.name, {'long-option': True}, _long_prefix='-custom-').strip()
        self.assertEqual(out, '-custom-long-option')
        out = python.bake(py.name, {'long-option': 'underscore'}, _long_prefix='-baked-')().strip()
        self.assertEqual(out, '-baked-long-option=underscore')
        out = python.bake(py.name, {'long-option': True}, _long_prefix='-baked-')().strip()
        self.assertEqual(out, '-baked-long-option')

    def test_command_wrapper(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import Command, which
        ls = Command(str(which('ls')).strip())
        wc = Command(str(which('wc')).strip())
        c1 = int(wc(l=True, _in=ls('-A1', THIS_DIR, _return_cmd=True)))
        c2 = len(os.listdir(THIS_DIR))
        self.assertEqual(c1, c2)

    def test_background(self):
        if False:
            print('Hello World!')
        import time
        from sh import sleep
        start = time.time()
        sleep_time = 0.5
        p = sleep(sleep_time, _bg=True)
        now = time.time()
        self.assertLess(now - start, sleep_time)
        p.wait()
        now = time.time()
        self.assertGreater(now - start, sleep_time)

    def test_background_exception(self):
        if False:
            return 10
        py = create_tmp_test('exit(1)')
        p = python(py.name, _bg=True, _bg_exc=False)
        self.assertRaises(sh.ErrorReturnCode_1, p.wait)

    def test_with_context(self):
        if False:
            return 10
        import getpass
        from sh import whoami
        py = create_tmp_test('\nimport sys\nimport os\nimport subprocess\n\nprint("with_context")\nsubprocess.Popen(sys.argv[1:], shell=False).wait()\n')
        cmd1 = python.bake(py.name, _with=True)
        with cmd1:
            out = whoami()
        self.assertIn('with_context', out)
        self.assertIn(getpass.getuser(), out)

    def test_with_context_args(self):
        if False:
            for i in range(10):
                print('nop')
        import getpass
        from sh import whoami
        py = create_tmp_test('\nimport sys\nimport os\nimport subprocess\nfrom optparse import OptionParser\n\nparser = OptionParser()\nparser.add_option("-o", "--opt", action="store_true", default=False, dest="opt")\noptions, args = parser.parse_args()\n\nif options.opt:\n    subprocess.Popen(args[0], shell=False).wait()\n')
        with python(py.name, opt=True, _with=True):
            out = whoami()
        self.assertEqual(getpass.getuser(), out.strip())
        with python(py.name, _with=True):
            out = whoami()
        self.assertEqual(out.strip(), '')

    def test_with_context_nested(self):
        if False:
            while True:
                i = 10
        echo_path = sh.echo._path
        with sh.echo.bake('test1', _with=True):
            with sh.echo.bake('test2', _with=True):
                out = sh.echo('test3')
        self.assertEqual(out.strip(), f'test1 {echo_path} test2 {echo_path} test3')

    def test_binary_input(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nimport sys\ndata = sys.stdin.read()\nsys.stdout.write(data)\n')
        data = b'1234'
        out = pythons(py.name, _in=data)
        self.assertEqual(out, '1234')

    def test_err_to_out(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stdout.flush()\nsys.stderr.write("stderr")\nsys.stderr.flush()\n')
        stdout = pythons(py.name, _err_to_out=True)
        self.assertEqual(stdout, 'stdoutstderr')

    def test_err_to_out_and_sys_stdout(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stdout.flush()\nsys.stderr.write("stderr")\nsys.stderr.flush()\n')
        (master, slave) = os.pipe()
        stdout = pythons(py.name, _err_to_out=True, _out=slave)
        self.assertEqual(stdout, '')
        self.assertEqual(os.read(master, 12), b'stdoutstderr')

    def test_err_piped(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nsys.stderr.write("stderr")\n')
        py2 = create_tmp_test('\nimport sys\nwhile True:\n    line = sys.stdin.read()\n    if not line:\n        break\n    sys.stdout.write(line)\n')
        out = pythons('-u', py2.name, _in=python('-u', py.name, _piped='err'))
        self.assertEqual(out, 'stderr')

    def test_out_redirection(self):
        if False:
            print('Hello World!')
        import tempfile
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        file_obj = tempfile.NamedTemporaryFile()
        out = python(py.name, _out=file_obj)
        self.assertEqual(len(out), 0)
        file_obj.seek(0)
        actual_out = file_obj.read()
        file_obj.close()
        self.assertNotEqual(len(actual_out), 0)
        file_obj = tempfile.NamedTemporaryFile()
        out = python(py.name, _out=file_obj, _tee=True)
        self.assertGreater(len(out), 0)
        file_obj.seek(0)
        actual_out = file_obj.read()
        file_obj.close()
        self.assertGreater(len(actual_out), 0)

    def test_err_redirection(self):
        if False:
            while True:
                i = 10
        import tempfile
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        file_obj = tempfile.NamedTemporaryFile()
        p = python('-u', py.name, _err=file_obj)
        file_obj.seek(0)
        stderr = file_obj.read().decode()
        file_obj.close()
        self.assertEqual(p.stdout, b'stdout')
        self.assertEqual(stderr, 'stderr')
        self.assertEqual(len(p.stderr), 0)
        file_obj = tempfile.NamedTemporaryFile()
        p = python(py.name, _err=file_obj, _tee='err')
        file_obj.seek(0)
        stderr = file_obj.read().decode()
        file_obj.close()
        self.assertEqual(p.stdout, b'stdout')
        self.assertEqual(stderr, 'stderr')
        self.assertGreater(len(p.stderr), 0)

    def test_out_and_err_redirection(self):
        if False:
            i = 10
            return i + 15
        import tempfile
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        err_file_obj = tempfile.NamedTemporaryFile()
        out_file_obj = tempfile.NamedTemporaryFile()
        p = python(py.name, _out=out_file_obj, _err=err_file_obj, _tee=('err', 'out'))
        out_file_obj.seek(0)
        stdout = out_file_obj.read().decode()
        out_file_obj.close()
        err_file_obj.seek(0)
        stderr = err_file_obj.read().decode()
        err_file_obj.close()
        self.assertEqual(stdout, 'stdout')
        self.assertEqual(p.stdout, b'stdout')
        self.assertEqual(stderr, 'stderr')
        self.assertEqual(p.stderr, b'stderr')

    def test_tty_tee(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write("stdout")\n')
        (read, write) = pty.openpty()
        out = python('-u', py.name, _out=write).stdout
        tee = os.read(read, 6)
        self.assertEqual(out, b'')
        self.assertEqual(tee, b'stdout')
        os.close(write)
        os.close(read)
        (read, write) = pty.openpty()
        out = python('-u', py.name, _out=write, _tee=True).stdout
        tee = os.read(read, 6)
        self.assertEqual(out, b'stdout')
        self.assertEqual(tee, b'stdout')
        os.close(write)
        os.close(read)

    def test_err_redirection_actual_file(self):
        if False:
            for i in range(10):
                print('nop')
        import tempfile
        file_obj = tempfile.NamedTemporaryFile()
        py = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        stdout = pythons('-u', py.name, _err=file_obj.name)
        file_obj.seek(0)
        stderr = file_obj.read().decode()
        file_obj.close()
        self.assertEqual(stdout, 'stdout')
        self.assertEqual(stderr, 'stderr')

    def test_subcommand_and_bake(self):
        if False:
            print('Hello World!')
        import getpass
        py = create_tmp_test('\nimport sys\nimport os\nimport subprocess\n\nprint("subcommand")\nsubprocess.Popen(sys.argv[1:], shell=False).wait()\n')
        cmd1 = python.bake(py.name)
        out = cmd1.whoami()
        self.assertIn('subcommand', out)
        self.assertIn(getpass.getuser(), out)

    def test_multiple_bakes(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport sys\nsys.stdout.write(str(sys.argv[1:]))\n')
        out = python.bake(py.name).bake('bake1').bake('bake2')()
        self.assertEqual("['bake1', 'bake2']", str(out))

    def test_arg_preprocessor(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nimport sys\nsys.stdout.write(str(sys.argv[1:]))\n')

        def arg_preprocess(args, kwargs):
            if False:
                i = 10
                return i + 15
            args.insert(0, 'preprocessed')
            kwargs['a-kwarg'] = 123
            return (args, kwargs)
        cmd = pythons.bake(py.name, _arg_preprocess=arg_preprocess)
        out = cmd('arg')
        self.assertEqual("['preprocessed', 'arg', '--a-kwarg=123']", out)

    def test_bake_args_come_first(self):
        if False:
            return 10
        from sh import ls
        ls = ls.bake(h=True)
        ran = ls('-la', _return_cmd=True).ran
        ft = ran.index('-h')
        self.assertIn('-la', ran[ft:])

    def test_output_equivalence(self):
        if False:
            return 10
        from sh import whoami
        iam1 = whoami()
        iam2 = whoami()
        self.assertEqual(iam1, iam2)

    def test_stdout_pipe(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\n\nsys.stdout.write("foobar\\n")\n')
        (read_fd, write_fd) = os.pipe()
        python(py.name, _out=write_fd, u=True)

        def alarm(sig, action):
            if False:
                return 10
            self.fail('Timeout while reading from pipe')
        import signal
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(3)
        data = os.read(read_fd, 100)
        self.assertEqual(b'foobar\n', data)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def test_stdout_callback(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): print(i)\n')
        stdout = []

        def agg(line):
            if False:
                i = 10
                return i + 15
            stdout.append(line)
        p = python('-u', py.name, _out=agg)
        p.wait()
        self.assertEqual(len(stdout), 5)

    def test_stdout_callback_no_wait(self):
        if False:
            print('Hello World!')
        import time
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor i in range(5):\n    print(i)\n    time.sleep(.5)\n')
        stdout = []

        def agg(line):
            if False:
                return 10
            stdout.append(line)
        python('-u', py.name, _out=agg, _bg=True)
        time.sleep(0.5)
        self.assertNotEqual(len(stdout), 5)

    def test_stdout_callback_line_buffered(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): print("herpderp")\n')
        stdout = []

        def agg(line):
            if False:
                return 10
            stdout.append(line)
        p = python('-u', py.name, _out=agg, _out_bufsize=1)
        p.wait()
        self.assertEqual(len(stdout), 5)

    def test_stdout_callback_line_unbuffered(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): print("herpderp")\n')
        stdout = []

        def agg(char):
            if False:
                i = 10
                return i + 15
            stdout.append(char)
        p = python('-u', py.name, _out=agg, _out_bufsize=0)
        p.wait()
        self.assertEqual(len(stdout), len('herpderp') * 5 + 5)

    def test_stdout_callback_buffered(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): sys.stdout.write("herpderp")\n')
        stdout = []

        def agg(chunk):
            if False:
                i = 10
                return i + 15
            stdout.append(chunk)
        p = python('-u', py.name, _out=agg, _out_bufsize=4)
        p.wait()
        self.assertEqual(len(stdout), len('herp') / 2 * 5)

    def test_stdout_callback_with_input(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): print(str(i))\nderp = input("herp? ")\nprint(derp)\n')

        def agg(line, stdin):
            if False:
                while True:
                    i = 10
            if line.strip() == '4':
                stdin.put('derp\n')
        p = python('-u', py.name, _out=agg, _tee=True)
        p.wait()
        self.assertIn('derp', p)

    def test_stdout_callback_exit(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(5): print(i)\n')
        stdout = []

        def agg(line):
            if False:
                i = 10
                return i + 15
            line = line.strip()
            stdout.append(line)
            if line == '2':
                return True
        p = python('-u', py.name, _out=agg, _tee=True)
        p.wait()
        self.assertIn('4', p)
        self.assertNotIn('4', stdout)

    def test_stdout_callback_terminate(self):
        if False:
            print('Hello World!')
        import signal
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor i in range(5):\n    print(i)\n    time.sleep(.5)\n')
        stdout = []

        def agg(line, stdin, process):
            if False:
                i = 10
                return i + 15
            line = line.strip()
            stdout.append(line)
            if line == '3':
                process.terminate()
                return True
        import sh
        caught_signal = False
        try:
            p = python('-u', py.name, _out=agg, _bg=True)
            p.wait()
        except sh.SignalException_SIGTERM:
            caught_signal = True
        self.assertTrue(caught_signal)
        self.assertEqual(p.process.exit_code, -signal.SIGTERM)
        self.assertNotIn('4', p)
        self.assertNotIn('4', stdout)

    def test_stdout_callback_kill(self):
        if False:
            return 10
        import signal
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor i in range(5):\n    print(i)\n    time.sleep(.5)\n')
        stdout = []

        def agg(line, stdin, process):
            if False:
                return 10
            line = line.strip()
            stdout.append(line)
            if line == '3':
                process.kill()
                return True
        import sh
        caught_signal = False
        try:
            p = python('-u', py.name, _out=agg, _bg=True)
            p.wait()
        except sh.SignalException_SIGKILL:
            caught_signal = True
        self.assertTrue(caught_signal)
        self.assertEqual(p.process.exit_code, -signal.SIGKILL)
        self.assertNotIn('4', p)
        self.assertNotIn('4', stdout)

    def test_general_signal(self):
        if False:
            i = 10
            return i + 15
        from signal import SIGINT
        py = create_tmp_test('\nimport sys\nimport os\nimport time\nimport signal\n\ni = 0\ndef sig_handler(sig, frame):\n    global i\n    i = 42\n\nsignal.signal(signal.SIGINT, sig_handler)\n\nfor _ in range(6):\n    print(i)\n    i += 1\n    sys.stdout.flush()\n    time.sleep(2)\n')
        stdout = []

        def agg(line, stdin, process):
            if False:
                for i in range(10):
                    print('nop')
            line = line.strip()
            stdout.append(line)
            if line == '3':
                process.signal(SIGINT)
                return True
        p = python(py.name, _out=agg, _tee=True)
        p.wait()
        self.assertEqual(p.process.exit_code, 0)
        self.assertEqual(str(p), '0\n1\n2\n3\n42\n43\n')

    def test_iter_generator(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor i in range(42):\n    print(i)\n    sys.stdout.flush()\n')
        out = []
        for line in python(py.name, _iter=True):
            out.append(int(line.strip()))
        self.assertEqual(len(out), 42)
        self.assertEqual(sum(out), 861)

    def test_async(self):
        if False:
            return 10
        py = create_tmp_test('\nimport os\nimport time\ntime.sleep(0.5)\nprint("hello")\n')
        alternating = []
        q = AQueue()

        async def producer(q):
            alternating.append(1)
            msg = await python(py.name, _async=True)
            alternating.append(1)
            await q.put(msg.strip())

        async def consumer(q):
            await asyncio.sleep(0.1)
            alternating.append(2)
            msg = await q.get()
            self.assertEqual(msg, 'hello')
            alternating.append(2)
        loop = asyncio.get_event_loop()
        fut = asyncio.gather(producer(q), consumer(q))
        loop.run_until_complete(fut)
        self.assertListEqual(alternating, [1, 2, 1, 2])

    def test_async_exc(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('exit(34)')

        async def producer():
            await python(py.name, _async=True)
        loop = asyncio.get_event_loop()
        self.assertRaises(sh.ErrorReturnCode_34, loop.run_until_complete, producer())

    def test_async_iter(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nfor i in range(5):\n    print(i)\n')
        q = AQueue()
        alternating = []

        async def producer(q):
            async for line in python(py.name, _iter=True):
                alternating.append(1)
                await q.put(int(line.strip()))
            await q.put(None)

        async def consumer(q):
            while True:
                line = await q.get()
                if line is None:
                    return
                alternating.append(2)
        loop = asyncio.get_event_loop()
        res = asyncio.gather(producer(q), consumer(q))
        loop.run_until_complete(res)
        self.assertListEqual(alternating, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

    def test_async_iter_exc(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nfor i in range(5):\n    print(i)\nexit(34)\n')
        lines = []

        async def producer():
            async for line in python(py.name, _async=True):
                lines.append(int(line.strip()))
        loop = asyncio.get_event_loop()
        self.assertRaises(sh.ErrorReturnCode_34, loop.run_until_complete, producer())

    def test_handle_both_out_and_err(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor i in range(42):\n    sys.stdout.write(str(i) + "\\n")\n    sys.stdout.flush()\n    if i % 2 == 0:\n        sys.stderr.write(str(i) + "\\n")\n        sys.stderr.flush()\n')
        out = []

        def handle_out(line):
            if False:
                print('Hello World!')
            out.append(int(line.strip()))
        err = []

        def handle_err(line):
            if False:
                return 10
            err.append(int(line.strip()))
        p = python(py.name, _err=handle_err, _out=handle_out, _bg=True)
        p.wait()
        self.assertEqual(sum(out), 861)
        self.assertEqual(sum(err), 420)

    def test_iter_unicode(self):
        if False:
            return 10
        test_string = 'ä½\x95ä½\x95\n' * 150
        txt = create_tmp_test(test_string)
        for line in sh.cat(txt.name, _iter=True):
            break
        self.assertLess(len(line), 1024)

    def test_nonblocking_iter(self):
        if False:
            for i in range(10):
                print('nop')
        from errno import EWOULDBLOCK
        py = create_tmp_test('\nimport time\nimport sys\ntime.sleep(1)\nsys.stdout.write("stdout")\n')
        count = 0
        value = None
        for line in python(py.name, _iter_noblock=True):
            if line == EWOULDBLOCK:
                count += 1
            else:
                value = line
        self.assertGreater(count, 0)
        self.assertEqual(value, 'stdout')
        py = create_tmp_test('\nimport time\nimport sys\ntime.sleep(1)\nsys.stderr.write("stderr")\n')
        count = 0
        value = None
        for line in python(py.name, _iter_noblock='err'):
            if line == EWOULDBLOCK:
                count += 1
            else:
                value = line
        self.assertGreater(count, 0)
        self.assertEqual(value, 'stderr')

    def test_for_generator_to_err(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(42):\n    sys.stderr.write(str(i)+"\\n")\n')
        out = []
        for line in python('-u', py.name, _iter='err'):
            out.append(line)
        self.assertEqual(len(out), 42)
        out = []
        for line in python('-u', py.name, _iter='out'):
            out.append(line)
        self.assertEqual(len(out), 0)

    def test_sigpipe(self):
        if False:
            for i in range(10):
                print('nop')
        py1 = create_tmp_test('\nimport sys\nimport os\nimport time\nimport signal\n\n# by default, python disables SIGPIPE, in favor of using IOError exceptions, so\n# let\'s put that back to the system default where we terminate with a signal\n# exit code\nsignal.signal(signal.SIGPIPE, signal.SIG_DFL)\n\nfor letter in "andrew":\n    time.sleep(0.6)\n    print(letter)\n        ')
        py2 = create_tmp_test('\nimport sys\nimport os\nimport time\n\nwhile True:\n    line = sys.stdin.readline()\n    if not line:\n        break\n    print(line.strip().upper())\n    exit(0)\n        ')
        p1 = python('-u', py1.name, _piped='out')
        p2 = python('-u', py2.name, _in=p1)
        self.assertEqual(-p1.exit_code, signal.SIGPIPE)
        self.assertEqual(p2.exit_code, 0)

    def test_piped_generator(self):
        if False:
            return 10
        import time
        py1 = create_tmp_test('\nimport sys\nimport os\nimport time\n\nfor letter in "andrew":\n    time.sleep(0.6)\n    print(letter)\n        ')
        py2 = create_tmp_test('\nimport sys\nimport os\nimport time\n\nwhile True:\n    line = sys.stdin.readline()\n    if not line:\n        break\n    print(line.strip().upper())\n        ')
        times = []
        last_received = None
        letters = ''
        for line in python('-u', py2.name, _iter=True, _in=python('-u', py1.name, _piped='out')):
            letters += line.strip()
            now = time.time()
            if last_received:
                times.append(now - last_received)
            last_received = now
        self.assertEqual('ANDREW', letters)
        self.assertTrue(all([t > 0.3 for t in times]))

    def test_no_out_iter_err(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nsys.stderr.write("1\\n")\nsys.stderr.write("2\\n")\nsys.stderr.write("3\\n")\nsys.stderr.flush()\n')
        nums = [int(num.strip()) for num in python(py.name, _iter='err', _no_out=True)]
        assert nums == [1, 2, 3]

    def test_generator_and_callback(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nimport sys\nimport os\n\nfor i in range(42):\n    sys.stderr.write(str(i * 2)+"\\n")\n    print(i)\n')
        stderr = []

        def agg(line):
            if False:
                print('Hello World!')
            stderr.append(int(line.strip()))
        out = []
        for line in python('-u', py.name, _iter=True, _err=agg):
            out.append(line)
        self.assertEqual(len(out), 42)
        self.assertEqual(sum(stderr), 1722)

    def test_cast_bg(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport sys\nimport time\ntime.sleep(0.5)\nsys.stdout.write(sys.argv[1])\n')
        self.assertEqual(int(python(py.name, '123', _bg=True)), 123)
        self.assertEqual(float(python(py.name, '789', _bg=True)), 789.0)

    def test_cmd_eq(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('')
        cmd1 = python.bake(py.name, '-u')
        cmd2 = python.bake(py.name, '-u')
        cmd3 = python.bake(py.name)
        self.assertEqual(cmd1, cmd2)
        self.assertNotEqual(cmd1, cmd3)

    def test_fg(self):
        if False:
            return 10
        py = create_tmp_test('exit(0)')
        system_python(py.name, _fg=True)

    def test_fg_false(self):
        if False:
            while True:
                i = 10
        'https://github.com/amoffat/sh/issues/520'
        py = create_tmp_test("print('hello')")
        buf = StringIO()
        python(py.name, _fg=False, _out=buf)
        self.assertEqual(buf.getvalue(), 'hello\n')

    def test_fg_true(self):
        if False:
            return 10
        'https://github.com/amoffat/sh/issues/520'
        py = create_tmp_test("print('hello')")
        buf = StringIO()
        self.assertRaises(TypeError, python, py.name, _fg=True, _out=buf)

    def test_fg_env(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nimport os\ncode = int(os.environ.get("EXIT", "0"))\nexit(code)\n')
        env = os.environ.copy()
        env['EXIT'] = '3'
        self.assertRaises(sh.ErrorReturnCode_3, python, py.name, _fg=True, _env=env)

    def test_fg_alternative(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('exit(0)')
        python(py.name, _in=sys.stdin, _out=sys.stdout, _err=sys.stderr)

    def test_fg_exc(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('exit(1)')
        self.assertRaises(sh.ErrorReturnCode_1, python, py.name, _fg=True)

    def test_out_filename(self):
        if False:
            while True:
                i = 10
        outfile = tempfile.NamedTemporaryFile()
        py = create_tmp_test("print('output')")
        python(py.name, _out=outfile.name)
        outfile.seek(0)
        self.assertEqual(b'output\n', outfile.read())

    def test_out_pathlike(self):
        if False:
            return 10
        from pathlib import Path
        outfile = tempfile.NamedTemporaryFile()
        py = create_tmp_test("print('output')")
        python(py.name, _out=Path(outfile.name))
        outfile.seek(0)
        self.assertEqual(b'output\n', outfile.read())

    def test_bg_exit_code(self):
        if False:
            return 10
        py = create_tmp_test('\nimport time\ntime.sleep(1)\nexit(49)\n')
        p = python(py.name, _ok_code=49, _bg=True)
        self.assertEqual(49, p.exit_code)

    def test_cwd(self):
        if False:
            i = 10
            return i + 15
        from os.path import realpath
        from sh import pwd
        self.assertEqual(str(pwd(_cwd='/tmp')), realpath('/tmp') + '\n')
        self.assertEqual(str(pwd(_cwd='/etc')), realpath('/etc') + '\n')

    def test_cwd_fg(self):
        if False:
            i = 10
            return i + 15
        td = realpath(tempfile.mkdtemp())
        py = create_tmp_test(f'\nimport sh\nimport os\nfrom os.path import realpath\norig = realpath(os.getcwd())\nprint(orig)\nsh.pwd(_cwd="{td}", _fg=True)\nprint(realpath(os.getcwd()))\n')
        (orig, newdir, restored) = python(py.name).strip().split('\n')
        newdir = realpath(newdir)
        self.assertEqual(newdir, td)
        self.assertEqual(orig, restored)
        self.assertNotEqual(orig, newdir)
        os.rmdir(td)

    def test_huge_piped_data(self):
        if False:
            print('Hello World!')
        from sh import tr
        stdin = tempfile.NamedTemporaryFile()
        data = 'herpderp' * 4000 + '\n'
        stdin.write(data.encode())
        stdin.flush()
        stdin.seek(0)
        out = tr('[:upper:]', '[:lower:]', _in=tr('[:lower:]', '[:upper:]', _in=data))
        self.assertTrue(out == data)

    def test_tty_input(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\nif os.isatty(sys.stdin.fileno()):\n    sys.stdout.write("password?\\n")\n    sys.stdout.flush()\n    pw = sys.stdin.readline().strip()\n    sys.stdout.write("%s\\n" % ("*" * len(pw)))\n    sys.stdout.flush()\nelse:\n    sys.stdout.write("no tty attached!\\n")\n    sys.stdout.flush()\n')
        test_pw = 'test123'
        expected_stars = '*' * len(test_pw)
        d = {}

        def password_enterer(line, stdin):
            if False:
                return 10
            line = line.strip()
            if not line:
                return
            if line == 'password?':
                stdin.put(test_pw + '\n')
            elif line.startswith('*'):
                d['stars'] = line
                return True
        pw_stars = python(py.name, _tty_in=True, _out=password_enterer)
        pw_stars.wait()
        self.assertEqual(d['stars'], expected_stars)
        response = python(py.name)
        self.assertEqual(str(response), 'no tty attached!\n')

    def test_tty_output(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nimport sys\nimport os\n\nif os.isatty(sys.stdout.fileno()):\n    sys.stdout.write("tty attached")\n    sys.stdout.flush()\nelse:\n    sys.stdout.write("no tty attached")\n    sys.stdout.flush()\n')
        out = pythons(py.name, _tty_out=True)
        self.assertEqual(out, 'tty attached')
        out = pythons(py.name, _tty_out=False)
        self.assertEqual(out, 'no tty attached')

    def test_stringio_output(self):
        if False:
            for i in range(10):
                print('nop')
        import sh
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.argv[1])\n')
        out = StringIO()
        sh.python(py.name, 'testing 123', _out=out)
        self.assertEqual(out.getvalue(), 'testing 123')
        out = BytesIO()
        sh.python(py.name, 'testing 123', _out=out)
        self.assertEqual(out.getvalue().decode(), 'testing 123')

    def test_stringio_input(self):
        if False:
            i = 10
            return i + 15
        from sh import cat
        input = StringIO()
        input.write('herpderp')
        input.seek(0)
        out = cat(_in=input)
        self.assertEqual(out, 'herpderp')

    def test_internal_bufsize(self):
        if False:
            return 10
        from sh import cat
        output = cat(_in='a' * 1000, _internal_bufsize=100, _out_bufsize=0)
        self.assertEqual(len(output), 100)
        output = cat(_in='a' * 1000, _internal_bufsize=50, _out_bufsize=2)
        self.assertEqual(len(output), 100)

    def test_change_stdout_buffering(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nimport os\n\n# this proves that we won\'t get the output into our callback until we send\n# a newline\nsys.stdout.write("switch ")\nsys.stdout.flush()\nsys.stdout.write("buffering\\n")\nsys.stdout.flush()\n\nsys.stdin.read(1)\nsys.stdout.write("unbuffered")\nsys.stdout.flush()\n\n# this is to keep the output from being flushed by the process ending, which\n# would ruin our test.  we want to make sure we get the string "unbuffered"\n# before the process ends, without writing a newline\nsys.stdin.read(1)\n')
        d = {'newline_buffer_success': False, 'unbuffered_success': False}

        def interact(line, stdin, process):
            if False:
                for i in range(10):
                    print('nop')
            line = line.strip()
            if not line:
                return
            if line == 'switch buffering':
                d['newline_buffer_success'] = True
                process.change_out_bufsize(0)
                stdin.put('a')
            elif line == 'unbuffered':
                stdin.put('b')
                d['unbuffered_success'] = True
                return True
        pw_stars = python('-u', py.name, _out=interact, _out_bufsize=1)
        pw_stars.wait()
        self.assertTrue(d['newline_buffer_success'])
        self.assertTrue(d['unbuffered_success'])

    def test_callable_interact(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write("line1")\n')

        class Callable(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.line = None

            def __call__(self, line):
                if False:
                    print('Hello World!')
                self.line = line
        cb = Callable()
        python(py.name, _out=cb)
        self.assertEqual(cb.line, 'line1')

    def test_encoding(self):
        if False:
            i = 10
            return i + 15
        return self.skipTest("what's the best way to test a different '_encoding' special keywordargument?")

    def test_timeout(self):
        if False:
            while True:
                i = 10
        from time import time
        import sh
        sleep_for = 3
        timeout = 1
        started = time()
        try:
            sh.sleep(sleep_for, _timeout=timeout).wait()
        except sh.TimeoutException as e:
            assert 'sleep 3' in e.full_cmd
        else:
            self.fail('no timeout exception')
        elapsed = time() - started
        self.assertLess(abs(elapsed - timeout), 0.5)

    def test_timeout_overstep(self):
        if False:
            i = 10
            return i + 15
        started = time.time()
        sh.sleep(1, _timeout=5)
        elapsed = time.time() - started
        self.assertLess(abs(elapsed - 1), 0.5)

    def test_timeout_wait(self):
        if False:
            print('Hello World!')
        p = sh.sleep(3, _bg=True)
        self.assertRaises(sh.TimeoutException, p.wait, timeout=1)

    def test_timeout_wait_overstep(self):
        if False:
            while True:
                i = 10
        p = sh.sleep(1, _bg=True)
        p.wait(timeout=5)

    def test_timeout_wait_negative(self):
        if False:
            i = 10
            return i + 15
        p = sh.sleep(3, _bg=True)
        self.assertRaises(RuntimeError, p.wait, timeout=-3)

    def test_binary_pipe(self):
        if False:
            while True:
                i = 10
        binary = b'\xec;\xedr\xdbF'
        py1 = create_tmp_test('\nimport sys\nimport os\n\nsys.stdout = os.fdopen(sys.stdout.fileno(), "wb", 0)\nsys.stdout.write(b\'\\xec;\\xedr\\xdbF\')\n')
        py2 = create_tmp_test('\nimport sys\nimport os\n\nsys.stdin = os.fdopen(sys.stdin.fileno(), "rb", 0)\nsys.stdout = os.fdopen(sys.stdout.fileno(), "wb", 0)\nsys.stdout.write(sys.stdin.read())\n')
        out = python(py2.name, _in=python(py1.name))
        self.assertEqual(out.stdout, binary)

    def test_failure_with_large_output(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import ErrorReturnCode_1
        py = create_tmp_test('\nprint("andrewmoffat" * 1000)\nexit(1)\n')
        self.assertRaises(ErrorReturnCode_1, python, py.name)

    def test_non_ascii_error(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import ErrorReturnCode, ls
        test = '/á'
        self.assertRaises(ErrorReturnCode, ls, test, _encoding='utf8')

    def test_no_out(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        p = python(py.name, _no_out=True)
        self.assertEqual(p.stdout, b'')
        self.assertEqual(p.stderr, b'stderr')
        self.assertTrue(p.process._pipe_queue.empty())

        def callback(line):
            if False:
                while True:
                    i = 10
            pass
        p = python(py.name, _out=callback)
        self.assertEqual(p.stdout, b'')
        self.assertEqual(p.stderr, b'stderr')
        self.assertTrue(p.process._pipe_queue.empty())
        p = python(py.name)
        self.assertEqual(p.stdout, b'stdout')
        self.assertEqual(p.stderr, b'stderr')
        self.assertFalse(p.process._pipe_queue.empty())

    def test_tty_stdin(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.stdin.read())\nsys.stdout.flush()\n')
        out = pythons(py.name, _in='test\n', _tty_in=True)
        self.assertEqual('test\n', out)

    def test_no_err(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write("stdout")\nsys.stderr.write("stderr")\n')
        p = python(py.name, _no_err=True)
        self.assertEqual(p.stderr, b'')
        self.assertEqual(p.stdout, b'stdout')
        self.assertFalse(p.process._pipe_queue.empty())

        def callback(line):
            if False:
                while True:
                    i = 10
            pass
        p = python(py.name, _err=callback)
        self.assertEqual(p.stderr, b'')
        self.assertEqual(p.stdout, b'stdout')
        self.assertFalse(p.process._pipe_queue.empty())
        p = python(py.name)
        self.assertEqual(p.stderr, b'stderr')
        self.assertEqual(p.stdout, b'stdout')
        self.assertFalse(p.process._pipe_queue.empty())

    def test_no_pipe(self):
        if False:
            return 10
        from sh import ls
        p = ls(_return_cmd=True)
        self.assertFalse(p.process._pipe_queue.empty())

        def callback(line):
            if False:
                print('Hello World!')
            pass
        p = ls(_out=callback, _return_cmd=True)
        self.assertTrue(p.process._pipe_queue.empty())
        p = ls(_no_pipe=True, _return_cmd=True)
        self.assertTrue(p.process._pipe_queue.empty())

    def test_decode_error_handling(self):
        if False:
            return 10
        from functools import partial
        py = create_tmp_test('\n# -*- coding: utf8 -*-\nimport sys\nimport os\nsys.stdout = os.fdopen(sys.stdout.fileno(), \'wb\')\nsys.stdout.write(bytes("te漢字st", "utf8") + "äåéë".encode("latin_1"))\n')
        fn = partial(pythons, py.name, _encoding='ascii')
        self.assertRaises(UnicodeDecodeError, fn)
        p = pythons(py.name, _encoding='ascii', _decode_errors='ignore')
        self.assertEqual(p, 'test')
        p = pythons(py.name, _encoding='ascii', _decode_errors='ignore', _out=sys.stdout, _tee=True)
        self.assertEqual(p, 'test')

    def test_signal_exception(self):
        if False:
            return 10
        from sh import SignalException_15

        def throw_terminate_signal():
            if False:
                for i in range(10):
                    print('nop')
            py = create_tmp_test('\nimport time\nwhile True: time.sleep(1)\n')
            to_kill = python(py.name, _bg=True)
            to_kill.terminate()
            to_kill.wait()
        self.assertRaises(SignalException_15, throw_terminate_signal)

    def test_signal_group(self):
        if False:
            return 10
        child = create_tmp_test('\nimport time\ntime.sleep(3)\n')
        parent = create_tmp_test('\nimport sys\nimport sh\npython = sh.Command(sys.executable)\np = python("{child_file}", _bg=True, _new_session=False)\nprint(p.pid)\nprint(p.process.pgid)\np.wait()\n', child_file=child.name)

        def launch():
            if False:
                while True:
                    i = 10
            p = python(parent.name, _bg=True, _iter=True, _new_group=True)
            child_pid = int(next(p).strip())
            child_pgid = int(next(p).strip())
            parent_pid = p.pid
            parent_pgid = p.process.pgid
            return (p, child_pid, child_pgid, parent_pid, parent_pgid)

        def assert_alive(pid):
            if False:
                while True:
                    i = 10
            os.kill(pid, 0)

        def assert_dead(pid):
            if False:
                return 10
            self.assert_oserror(errno.ESRCH, os.kill, pid, 0)
        (p, child_pid, child_pgid, parent_pid, parent_pgid) = launch()
        assert_alive(parent_pid)
        assert_alive(child_pid)
        p.kill()
        time.sleep(0.1)
        assert_dead(parent_pid)
        assert_alive(child_pid)
        self.assertRaises(sh.SignalException_SIGKILL, p.wait)
        assert_dead(child_pid)
        (p, child_pid, child_pgid, parent_pid, parent_pgid) = launch()
        assert_alive(parent_pid)
        assert_alive(child_pid)
        p.kill_group()
        time.sleep(0.1)
        assert_dead(parent_pid)
        assert_dead(child_pid)

    def test_pushd(self):
        if False:
            while True:
                i = 10
        'test basic pushd functionality'
        child = realpath(tempfile.mkdtemp())
        old_wd1 = sh.pwd().strip()
        old_wd2 = os.getcwd()
        self.assertEqual(old_wd1, old_wd2)
        self.assertNotEqual(old_wd1, child)
        with sh.pushd(child):
            new_wd1 = sh.pwd().strip()
            new_wd2 = os.getcwd()
        old_wd3 = sh.pwd().strip()
        old_wd4 = os.getcwd()
        self.assertEqual(old_wd3, old_wd4)
        self.assertEqual(old_wd1, old_wd3)
        self.assertEqual(new_wd1, child)
        self.assertEqual(new_wd2, child)

    def test_pushd_cd(self):
        if False:
            i = 10
            return i + 15
        'test that pushd works like pushd/popd'
        child = realpath(tempfile.mkdtemp())
        try:
            old_wd = os.getcwd()
            with sh.pushd(tempdir):
                self.assertEqual(str(tempdir), os.getcwd())
            self.assertEqual(old_wd, os.getcwd())
        finally:
            os.rmdir(child)

    def test_non_existant_cwd(self):
        if False:
            i = 10
            return i + 15
        from sh import ls
        non_exist_dir = join(tempdir, 'aowjgoahewro')
        self.assertFalse(exists(non_exist_dir))
        self.assertRaises(sh.ForkException, ls, _cwd=non_exist_dir)

    def test_baked_command_can_be_printed(self):
        if False:
            i = 10
            return i + 15
        from sh import ls
        ll = ls.bake('-l')
        self.assertTrue(str(ll).endswith('/ls -l'))

    def test_done_callback(self):
        if False:
            print('Hello World!')
        import time

        class Callback(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.called = False
                self.exit_code = None
                self.success = None

            def __call__(self, p, success, exit_code):
                if False:
                    return 10
                self.called = True
                self.exit_code = exit_code
                self.success = success
        py = create_tmp_test('\nfrom time import time, sleep\nsleep(1)\nprint(time())\n')
        callback = Callback()
        p = python(py.name, _done=callback, _bg=True)
        wait_start = time.time()
        p.wait()
        wait_elapsed = time.time() - wait_start
        self.assertTrue(callback.called)
        self.assertLess(abs(wait_elapsed - 1.0), 1.0)
        self.assertEqual(callback.exit_code, 0)
        self.assertTrue(callback.success)

    def test_done_callback_no_deadlock(self):
        if False:
            return 10
        import time
        py = create_tmp_test("\nfrom sh import sleep\n\ndef done(cmd, success, exit_code):\n    print(cmd, success, exit_code)\n\nsleep('1', _done=done)\n")
        p = python(py.name, _bg=True, _timeout=2)
        wait_start = time.time()
        p.wait()
        wait_elapsed = time.time() - wait_start
        self.assertLess(abs(wait_elapsed - 1.0), 1.0)

    def test_fork_exc(self):
        if False:
            print('Hello World!')
        from sh import ForkException
        py = create_tmp_test('')

        def fail():
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('nooo')
        self.assertRaises(ForkException, python, py.name, _preexec_fn=fail)

    def test_new_session_new_group(self):
        if False:
            i = 10
            return i + 15
        from threading import Event
        py = create_tmp_test('\nimport os\nimport time\npid = os.getpid()\npgid = os.getpgid(pid)\nsid = os.getsid(pid)\nstuff = [pid, pgid, sid]\n\nprint(",".join([str(el) for el in stuff]))\ntime.sleep(0.5)\n')
        event = Event()

        def handle(run_asserts, line, stdin, p):
            if False:
                i = 10
                return i + 15
            (pid, pgid, sid) = line.strip().split(',')
            pid = int(pid)
            pgid = int(pgid)
            sid = int(sid)
            test_pid = os.getpgid(os.getpid())
            self.assertEqual(p.pid, pid)
            self.assertEqual(p.pgid, pgid)
            self.assertEqual(pgid, p.get_pgid())
            self.assertEqual(p.sid, sid)
            self.assertEqual(sid, p.get_sid())
            run_asserts(pid, pgid, sid, test_pid)
            event.set()

        def session_true_group_false(pid, pgid, sid, test_pid):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(pid, sid)
            self.assertEqual(pid, pgid)
        p = python(py.name, _out=partial(handle, session_true_group_false), _new_session=True)
        p.wait()
        self.assertTrue(event.is_set())
        event.clear()

        def session_false_group_false(pid, pgid, sid, test_pid):
            if False:
                i = 10
                return i + 15
            self.assertEqual(test_pid, pgid)
            self.assertNotEqual(pid, sid)
        p = python(py.name, _out=partial(handle, session_false_group_false), _new_session=False)
        p.wait()
        self.assertTrue(event.is_set())
        event.clear()

        def session_false_group_true(pid, pgid, sid, test_pid):
            if False:
                print('Hello World!')
            self.assertEqual(pid, pgid)
            self.assertNotEqual(pid, sid)
        p = python(py.name, _out=partial(handle, session_false_group_true), _new_session=False, _new_group=True)
        p.wait()
        self.assertTrue(event.is_set())
        event.clear()

    def test_done_cb_exc(self):
        if False:
            i = 10
            return i + 15
        from sh import ErrorReturnCode

        class Callback(object):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.called = False
                self.success = None

            def __call__(self, p, success, exit_code):
                if False:
                    return 10
                self.success = success
                self.called = True
        py = create_tmp_test('exit(1)')
        callback = Callback()
        try:
            p = python(py.name, _done=callback, _bg=True)
            p.wait()
        except ErrorReturnCode:
            self.assertTrue(callback.called)
            self.assertFalse(callback.success)
        else:
            self.fail("command should've thrown an exception")

    def test_callable_stdin(self):
        if False:
            return 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.stdin.read())\n')

        def create_stdin():
            if False:
                return 10
            state = {'count': 0}

            def stdin():
                if False:
                    for i in range(10):
                        print('nop')
                count = state['count']
                if count == 4:
                    return None
                state['count'] += 1
                return str(count)
            return stdin
        out = pythons(py.name, _in=create_stdin())
        self.assertEqual('0123', out)

    def test_stdin_unbuffered_bufsize(self):
        if False:
            for i in range(10):
                print('nop')
        from time import sleep
        py = create_tmp_test('\nimport sys\nfrom time import time\n\nstarted = time()\ndata = sys.stdin.read(len("testing"))\nwaited = time() - started\nsys.stdout.write(data + "\\n")\nsys.stdout.write(str(waited) + "\\n")\n\nstarted = time()\ndata = sys.stdin.read(len("done"))\nwaited = time() - started\nsys.stdout.write(data + "\\n")\nsys.stdout.write(str(waited) + "\\n")\n\nsys.stdout.flush()\n')

        def create_stdin():
            if False:
                return 10
            yield 'test'
            sleep(1)
            yield 'ing'
            sleep(1)
            yield 'done'
        out = python(py.name, _in=create_stdin(), _in_bufsize=0)
        (word1, time1, word2, time2, _) = out.split('\n')
        time1 = float(time1)
        time2 = float(time2)
        self.assertEqual(word1, 'testing')
        self.assertLess(abs(1 - time1), 0.5)
        self.assertEqual(word2, 'done')
        self.assertLess(abs(1 - time2), 0.5)

    def test_stdin_newline_bufsize(self):
        if False:
            return 10
        from time import sleep
        py = create_tmp_test('\nimport sys\nfrom time import time\n\nstarted = time()\ndata = sys.stdin.read(len("testing\\n"))\nwaited = time() - started\nsys.stdout.write(data)\nsys.stdout.write(str(waited) + "\\n")\n\nstarted = time()\ndata = sys.stdin.read(len("done\\n"))\nwaited = time() - started\nsys.stdout.write(data)\nsys.stdout.write(str(waited) + "\\n")\n\nsys.stdout.flush()\n')

        def create_stdin():
            if False:
                return 10
            yield 'test'
            sleep(1)
            yield 'ing\n'
            sleep(1)
            yield 'done\n'
        out = python(py.name, _in=create_stdin(), _in_bufsize=1)
        (word1, time1, word2, time2, _) = out.split('\n')
        time1 = float(time1)
        time2 = float(time2)
        self.assertEqual(word1, 'testing')
        self.assertLess(abs(1 - time1), 0.5)
        self.assertEqual(word2, 'done')
        self.assertLess(abs(1 - time2), 0.5)

    def test_custom_timeout_signal(self):
        if False:
            print('Hello World!')
        import signal
        from sh import TimeoutException
        py = create_tmp_test('\nimport time\ntime.sleep(3)\n')
        try:
            python(py.name, _timeout=1, _timeout_signal=signal.SIGHUP)
        except TimeoutException as e:
            self.assertEqual(e.exit_code, signal.SIGHUP)
        else:
            self.fail('we should have handled a TimeoutException')

    def test_append_stdout(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nimport sys\nnum = sys.stdin.read()\nsys.stdout.write(num)\n')
        append_file = tempfile.NamedTemporaryFile(mode='a+b')
        python(py.name, _in='1', _out=append_file)
        python(py.name, _in='2', _out=append_file)
        append_file.seek(0)
        output = append_file.read()
        self.assertEqual(b'12', output)

    def test_shadowed_subcommand(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.argv[1])\n')
        out = pythons.bake(py.name).bake_()
        self.assertEqual('bake', out)

    def test_no_proc_no_attr(self):
        if False:
            while True:
                i = 10
        py = create_tmp_test('')
        with python(py.name) as p:
            self.assertRaises(AttributeError, getattr, p, 'exit_code')

    def test_partially_applied_callback(self):
        if False:
            i = 10
            return i + 15
        from functools import partial
        py = create_tmp_test('\nfor i in range(10):\n    print(i)\n')
        output = []

        def fn(foo, line):
            if False:
                i = 10
                return i + 15
            output.append((foo, int(line.strip())))
        log_line = partial(fn, 'hello')
        python(py.name, _out=log_line)
        self.assertEqual(output, [('hello', i) for i in range(10)])
        output = []

        def fn(foo, line, stdin, proc):
            if False:
                return 10
            output.append((foo, int(line.strip())))
        log_line = partial(fn, 'hello')
        python(py.name, _out=log_line)
        self.assertEqual(output, [('hello', i) for i in range(10)])

    def test_grandchild_no_sighup(self):
        if False:
            return 10
        import time
        child = create_tmp_test('\nimport signal\nimport sys\nimport time\n\noutput_file = sys.argv[1]\nwith open(output_file, "w") as f:\n    def handle_sighup(signum, frame):\n        f.write("got signal %d" % signum)\n        sys.exit(signum)\n    signal.signal(signal.SIGHUP, handle_sighup)\n    time.sleep(2)\n    f.write("made it!\\n")\n')
        parent = create_tmp_test('\nimport os\nimport time\nimport sys\n\nchild_file = sys.argv[1]\noutput_file = sys.argv[2]\n\npython_name = os.path.basename(sys.executable)\nos.spawnlp(os.P_NOWAIT, python_name, python_name, child_file, output_file)\ntime.sleep(1) # give child a chance to set up\n')
        output_file = tempfile.NamedTemporaryFile(delete=True)
        python(parent.name, child.name, output_file.name)
        time.sleep(3)
        out = output_file.readlines()[0]
        self.assertEqual(out, b'made it!\n')

    def test_unchecked_producer_failure(self):
        if False:
            i = 10
            return i + 15
        from sh import ErrorReturnCode_2
        producer = create_tmp_test('\nimport sys\nfor i in range(10):\n    print(i)\nsys.exit(2)\n')
        consumer = create_tmp_test('\nimport sys\nfor line in sys.stdin:\n    pass\n')
        direct_pipe = python(producer.name, _piped=True)
        self.assertRaises(ErrorReturnCode_2, python, direct_pipe, consumer.name)

    def test_unchecked_pipeline_failure(self):
        if False:
            print('Hello World!')
        from sh import ErrorReturnCode_2
        producer = create_tmp_test('\nimport sys\nfor i in range(10):\n    print(i)\nsys.exit(2)\n')
        middleman = create_tmp_test('\nimport sys\nfor line in sys.stdin:\n    print("> " + line)\n')
        consumer = create_tmp_test('\nimport sys\nfor line in sys.stdin:\n    pass\n')
        producer_normal_pipe = python(producer.name, _piped=True)
        middleman_normal_pipe = python(middleman.name, _piped=True, _in=producer_normal_pipe)
        self.assertRaises(ErrorReturnCode_2, python, middleman_normal_pipe, consumer.name)

class MockTests(BaseTests):

    def test_patch_command_cls(self):
        if False:
            return 10

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            cmd = sh.Command('afowejfow')
            return cmd()

        @unittest.mock.patch('sh.Command')
        def test(Command):
            if False:
                i = 10
                return i + 15
            Command().return_value = 'some output'
            return fn()
        self.assertEqual(test(), 'some output')
        self.assertRaises(sh.CommandNotFound, fn)

    def test_patch_command(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            return sh.afowejfow()

        @unittest.mock.patch('sh.afowejfow', create=True)
        def test(cmd):
            if False:
                return 10
            cmd.return_value = 'some output'
            return fn()
        self.assertEqual(test(), 'some output')
        self.assertRaises(sh.CommandNotFound, fn)

class MiscTests(BaseTests):

    def test_pickling(self):
        if False:
            return 10
        import pickle
        py = create_tmp_test('\nimport sys\nsys.stdout.write("some output")\nsys.stderr.write("some error")\nexit(1)\n')
        try:
            python(py.name)
        except sh.ErrorReturnCode as e:
            restored = pickle.loads(pickle.dumps(e))
            self.assertEqual(restored.stdout, b'some output')
            self.assertEqual(restored.stderr, b'some error')
            self.assertEqual(restored.exit_code, 1)
        else:
            self.fail("Didn't get an exception")

    @requires_poller('poll')
    def test_fd_over_1024(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('print("hi world")')
        with ulimit(resource.RLIMIT_NOFILE, 2048):
            cutoff_fd = 1024
            pipes = []
            for i in range(cutoff_fd):
                (master, slave) = os.pipe()
                pipes.append((master, slave))
                if slave >= cutoff_fd:
                    break
            python(py.name)
            for (master, slave) in pipes:
                os.close(master)
                os.close(slave)

    def test_args_deprecated(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(DeprecationWarning, sh.args, _env={})

    def test_percent_doesnt_fail_logging(self):
        if False:
            i = 10
            return i + 15
        "test that a command name doesn't interfere with string formatting in\n        the internal loggers"
        py = create_tmp_test('\nprint("cool")\n')
        python(py.name, '%')
        python(py.name, '%%')
        python(py.name, '%%%')

    def test_pushd_thread_safety(self):
        if False:
            return 10
        import threading
        import time
        temp1 = realpath(tempfile.mkdtemp())
        temp2 = realpath(tempfile.mkdtemp())
        try:
            results = [None, None]

            def fn1():
                if False:
                    i = 10
                    return i + 15
                with sh.pushd(temp1):
                    time.sleep(0.2)
                    results[0] = realpath(os.getcwd())

            def fn2():
                if False:
                    return 10
                time.sleep(0.1)
                with sh.pushd(temp2):
                    results[1] = realpath(os.getcwd())
                    time.sleep(0.3)
            t1 = threading.Thread(name='t1', target=fn1)
            t2 = threading.Thread(name='t2', target=fn2)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertEqual(results, [temp1, temp2])
        finally:
            os.rmdir(temp1)
            os.rmdir(temp2)

    def test_stdin_nohang(self):
        if False:
            i = 10
            return i + 15
        py = create_tmp_test('\nprint("hi")\n')
        (read, write) = os.pipe()
        stdin = os.fdopen(read, 'r')
        python(py.name, _in=stdin)

    @requires_utf8
    def test_unicode_path(self):
        if False:
            for i in range(10):
                print('nop')
        from sh import Command
        python_name = os.path.basename(sys.executable)
        py = create_tmp_test(f'#!/usr/bin/env {python_name}\n# -*- coding: utf8 -*-\nprint("字")\n', prefix='字', delete=False)
        try:
            py.close()
            os.chmod(py.name, int(493))
            cmd = Command(py.name)
            str(cmd)
            repr(cmd)
            running = cmd(_return_cmd=True)
            str(running)
            repr(running)
            str(running.process)
            repr(running.process)
        finally:
            os.unlink(py.name)

    def test_wraps(self):
        if False:
            i = 10
            return i + 15
        from sh import ls
        wraps(ls)(lambda f: True)

    def test_signal_exception_aliases(self):
        if False:
            while True:
                i = 10
        'proves that signal exceptions with numbers and names are equivalent'
        import signal
        import sh
        sig_name = f'SignalException_{signal.SIGQUIT}'
        sig = getattr(sh, sig_name)
        from sh import SignalException_SIGQUIT
        self.assertEqual(sig, SignalException_SIGQUIT)

    def test_change_log_message(self):
        if False:
            print('Hello World!')
        py = create_tmp_test('\nprint("cool")\n')

        def log_msg(cmd, call_args, pid=None):
            if False:
                print('Hello World!')
            return 'Hi! I ran something'
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        logger = logging.getLogger('sh')
        logger.setLevel(logging.INFO)
        try:
            logger.addHandler(handler)
            python(py.name, 'meow', 'bark', _log_msg=log_msg)
        finally:
            logger.removeHandler(handler)
        loglines = buf.getvalue().split('\n')
        self.assertTrue(loglines, 'Log handler captured no messages?')
        self.assertTrue(loglines[0].startswith('Hi! I ran something'))

    def test_stop_iteration_doesnt_block(self):
        if False:
            print('Hello World!')
        "proves that calling calling next() on a stopped iterator doesn't\n        hang."
        py = create_tmp_test('\nprint("cool")\n')
        p = python(py.name, _iter=True)
        for i in range(100):
            try:
                next(p)
            except StopIteration:
                pass

    def test_threaded_with_contexts(self):
        if False:
            for i in range(10):
                print('nop')
        import threading
        import time
        py = create_tmp_test('\nimport sys\na = sys.argv\nres = (a[1], a[3])\nsys.stdout.write(repr(res))\n')
        p1 = python.bake('-u', py.name, 1)
        p2 = python.bake('-u', py.name, 2)
        results = [None, None]

        def f1():
            if False:
                for i in range(10):
                    print('nop')
            with p1:
                time.sleep(1)
                results[0] = str(system_python('one'))

        def f2():
            if False:
                i = 10
                return i + 15
            with p2:
                results[1] = str(system_python('two'))
        t1 = threading.Thread(target=f1)
        t1.start()
        t2 = threading.Thread(target=f2)
        t2.start()
        t1.join()
        t2.join()
        correct = ["('1', 'one')", "('2', 'two')"]
        self.assertEqual(results, correct)

    def test_eintr(self):
        if False:
            while True:
                i = 10
        import signal

        def handler(num, frame):
            if False:
                return 10
            pass
        signal.signal(signal.SIGALRM, handler)
        py = create_tmp_test('\nimport time\ntime.sleep(2)\n')
        p = python(py.name, _bg=True)
        signal.alarm(1)
        p.wait()

class StreamBuffererTests(unittest.TestCase):

    def test_unbuffered(self):
        if False:
            i = 10
            return i + 15
        from sh import StreamBufferer
        b = StreamBufferer(0)
        self.assertEqual(b.process(b'test'), [b'test'])
        self.assertEqual(b.process(b'one'), [b'one'])
        self.assertEqual(b.process(b''), [b''])
        self.assertEqual(b.flush(), b'')

    def test_newline_buffered(self):
        if False:
            while True:
                i = 10
        from sh import StreamBufferer
        b = StreamBufferer(1)
        self.assertEqual(b.process(b'testing\none\ntwo'), [b'testing\n', b'one\n'])
        self.assertEqual(b.process(b'\nthree\nfour'), [b'two\n', b'three\n'])
        self.assertEqual(b.flush(), b'four')

    def test_chunk_buffered(self):
        if False:
            print('Hello World!')
        from sh import StreamBufferer
        b = StreamBufferer(10)
        self.assertEqual(b.process(b'testing\none\ntwo'), [b'testing\non'])
        self.assertEqual(b.process(b'\nthree\n'), [b'e\ntwo\nthre'])
        self.assertEqual(b.flush(), b'e\n')

@requires_posix
class ExecutionContextTests(unittest.TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        import sh
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.argv[1])\n')
        out = StringIO()
        sh2 = sh.bake(_out=out)
        sh2.python(py.name, 'TEST')
        self.assertEqual('TEST', out.getvalue())

    def test_multiline_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        py = create_tmp_test('\nimport os\nprint(os.environ["ABC"])\n')
        sh2 = sh.bake(_env={'ABC': '123'})
        output = sh2.python(py.name).strip()
        assert output == '123'

    def test_no_interfere1(self):
        if False:
            i = 10
            return i + 15
        import sh
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.argv[1])\n')
        out = StringIO()
        _sh = sh.bake(_out=out)
        _sh.python(py.name, 'TEST')
        self.assertEqual('TEST', out.getvalue())
        out.seek(0)
        out.truncate(0)
        sh.python(py.name, 'KO')
        self.assertEqual('', out.getvalue())

    def test_no_interfere2(self):
        if False:
            for i in range(10):
                print('nop')
        import sh
        out = StringIO()
        from sh import echo
        _sh = sh.bake(_out=out)
        echo('-n', 'TEST')
        self.assertEqual('', out.getvalue())

    def test_set_in_parent_function(self):
        if False:
            return 10
        import sh
        py = create_tmp_test('\nimport sys\nsys.stdout.write(sys.argv[1])\n')
        out = StringIO()
        _sh = sh.bake(_out=out)

        def nested1():
            if False:
                print('Hello World!')
            _sh.python(py.name, 'TEST1')

        def nested2():
            if False:
                print('Hello World!')
            import sh
            sh.python(py.name, 'TEST2')
        nested1()
        nested2()
        self.assertEqual('TEST1', out.getvalue())

    def test_command_with_baked_call_args(self):
        if False:
            i = 10
            return i + 15
        import sh
        _sh = sh.bake(_ok_code=1)
        self.assertEqual(sh.Command._call_args['ok_code'], 0)
        self.assertEqual(_sh.Command._call_args['ok_code'], 1)
if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(logging.NullHandler())
    test_kwargs = {'warnings': 'ignore'}
    if len(sys.argv) > 1:
        unittest.main(**test_kwargs)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        test_kwargs['verbosity'] = 2
        result = unittest.TextTestRunner(**test_kwargs).run(suite)
        if not result.wasSuccessful():
            exit(1)