import compileall
import contextlib
import filecmp
import importlib.util
import io
import os
import pathlib
import py_compile
import shutil
import struct
import sys
import tempfile
import test.test_importlib.util
import time
import unittest
from unittest import mock, skipUnless
from concurrent.futures import ProcessPoolExecutor
try:
    from concurrent.futures.process import _check_system_limits
    _check_system_limits()
    _have_multiprocessing = True
except NotImplementedError:
    _have_multiprocessing = False
from test import support
from test.support import os_helper
from test.support import script_helper
from test.test_py_compile import without_source_date_epoch
from test.test_py_compile import SourceDateEpochTestMeta

def get_pyc(script, opt):
    if False:
        for i in range(10):
            print('nop')
    if not opt:
        opt = ''
    return importlib.util.cache_from_source(script, optimization=opt)

def get_pycs(script):
    if False:
        return 10
    return [get_pyc(script, opt) for opt in (0, 1, 2)]

def is_hardlink(filename1, filename2):
    if False:
        while True:
            i = 10
    'Returns True if two files have the same inode (hardlink)'
    inode1 = os.stat(filename1).st_ino
    inode2 = os.stat(filename2).st_ino
    return inode1 == inode2

class CompileallTestsBase:

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.directory = tempfile.mkdtemp()
        self.source_path = os.path.join(self.directory, '_test.py')
        self.bc_path = importlib.util.cache_from_source(self.source_path)
        with open(self.source_path, 'w', encoding='utf-8') as file:
            file.write('x = 123\n')
        self.source_path2 = os.path.join(self.directory, '_test2.py')
        self.bc_path2 = importlib.util.cache_from_source(self.source_path2)
        shutil.copyfile(self.source_path, self.source_path2)
        self.subdirectory = os.path.join(self.directory, '_subdir')
        os.mkdir(self.subdirectory)
        self.source_path3 = os.path.join(self.subdirectory, '_test3.py')
        shutil.copyfile(self.source_path, self.source_path3)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.directory)

    def add_bad_source_file(self):
        if False:
            for i in range(10):
                print('nop')
        self.bad_source_path = os.path.join(self.directory, '_test_bad.py')
        with open(self.bad_source_path, 'w', encoding='utf-8') as file:
            file.write('x (\n')

    def timestamp_metadata(self):
        if False:
            return 10
        with open(self.bc_path, 'rb') as file:
            data = file.read(12)
        mtime = int(os.stat(self.source_path).st_mtime)
        compare = struct.pack('<4sLL', importlib.util.MAGIC_NUMBER, 0, mtime & 4294967295)
        return (data, compare)

    def test_year_2038_mtime_compilation(self):
        if False:
            return 10
        try:
            os.utime(self.source_path, (2 ** 32 - 1, 2 ** 32 - 1))
        except (OverflowError, OSError):
            self.skipTest("filesystem doesn't support timestamps near 2**32")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(compileall.compile_file(self.source_path))

    def test_larger_than_32_bit_times(self):
        if False:
            i = 10
            return i + 15
        try:
            os.utime(self.source_path, (2 ** 35, 2 ** 35))
        except (OverflowError, OSError):
            self.skipTest("filesystem doesn't support large timestamps")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(compileall.compile_file(self.source_path))

    def recreation_check(self, metadata):
        if False:
            return 10
        'Check that compileall recreates bytecode when the new metadata is\n        used.'
        if os.environ.get('SOURCE_DATE_EPOCH'):
            raise unittest.SkipTest('SOURCE_DATE_EPOCH is set')
        py_compile.compile(self.source_path)
        self.assertEqual(*self.timestamp_metadata())
        with open(self.bc_path, 'rb') as file:
            bc = file.read()[len(metadata):]
        with open(self.bc_path, 'wb') as file:
            file.write(metadata)
            file.write(bc)
        self.assertNotEqual(*self.timestamp_metadata())
        compileall.compile_dir(self.directory, force=False, quiet=True)
        self.assertTrue(*self.timestamp_metadata())

    def test_mtime(self):
        if False:
            i = 10
            return i + 15
        self.recreation_check(struct.pack('<4sLL', importlib.util.MAGIC_NUMBER, 0, 1))

    def test_magic_number(self):
        if False:
            for i in range(10):
                print('nop')
        self.recreation_check(b'\x00\x00\x00\x00')

    def test_compile_files(self):
        if False:
            return 10
        for fn in (self.bc_path, self.bc_path2):
            try:
                os.unlink(fn)
            except:
                pass
        self.assertTrue(compileall.compile_file(self.source_path, force=False, quiet=True))
        self.assertTrue(os.path.isfile(self.bc_path) and (not os.path.isfile(self.bc_path2)))
        os.unlink(self.bc_path)
        self.assertTrue(compileall.compile_dir(self.directory, force=False, quiet=True))
        self.assertTrue(os.path.isfile(self.bc_path) and os.path.isfile(self.bc_path2))
        os.unlink(self.bc_path)
        os.unlink(self.bc_path2)
        self.add_bad_source_file()
        self.assertFalse(compileall.compile_file(self.bad_source_path, force=False, quiet=2))
        self.assertFalse(compileall.compile_dir(self.directory, force=False, quiet=2))

    def test_compile_file_pathlike(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(os.path.isfile(self.bc_path))
        with support.captured_stdout() as stdout:
            self.assertTrue(compileall.compile_file(pathlib.Path(self.source_path)))
        self.assertRegex(stdout.getvalue(), 'Compiling ([^WindowsPath|PosixPath].*)')
        self.assertTrue(os.path.isfile(self.bc_path))

    def test_compile_file_pathlike_ddir(self):
        if False:
            while True:
                i = 10
        self.assertFalse(os.path.isfile(self.bc_path))
        self.assertTrue(compileall.compile_file(pathlib.Path(self.source_path), ddir=pathlib.Path('ddir_path'), quiet=2))
        self.assertTrue(os.path.isfile(self.bc_path))

    def test_compile_path(self):
        if False:
            while True:
                i = 10
        with test.test_importlib.util.import_state(path=[self.directory]):
            self.assertTrue(compileall.compile_path(quiet=2))
        with test.test_importlib.util.import_state(path=[self.directory]):
            self.add_bad_source_file()
            self.assertFalse(compileall.compile_path(skip_curdir=False, force=True, quiet=2))

    def test_no_pycache_in_non_package(self):
        if False:
            while True:
                i = 10
        data_dir = os.path.join(self.directory, 'data')
        data_file = os.path.join(data_dir, 'file')
        os.mkdir(data_dir)
        with open(data_file, 'wb'):
            pass
        compileall.compile_file(data_file)
        self.assertFalse(os.path.exists(os.path.join(data_dir, '__pycache__')))

    def test_compile_file_encoding_fallback(self):
        if False:
            print('Hello World!')
        self.add_bad_source_file()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(compileall.compile_file(self.bad_source_path))

    def test_optimize(self):
        if False:
            print('Hello World!')
        (optimize, opt) = (1, 1) if __debug__ else (0, '')
        compileall.compile_dir(self.directory, quiet=True, optimize=optimize)
        cached = importlib.util.cache_from_source(self.source_path, optimization=opt)
        self.assertTrue(os.path.isfile(cached))
        cached2 = importlib.util.cache_from_source(self.source_path2, optimization=opt)
        self.assertTrue(os.path.isfile(cached2))
        cached3 = importlib.util.cache_from_source(self.source_path3, optimization=opt)
        self.assertTrue(os.path.isfile(cached3))

    def test_compile_dir_pathlike(self):
        if False:
            print('Hello World!')
        self.assertFalse(os.path.isfile(self.bc_path))
        with support.captured_stdout() as stdout:
            compileall.compile_dir(pathlib.Path(self.directory))
        line = stdout.getvalue().splitlines()[0]
        self.assertRegex(line, 'Listing ([^WindowsPath|PosixPath].*)')
        self.assertTrue(os.path.isfile(self.bc_path))

    @skipUnless(_have_multiprocessing, 'requires multiprocessing')
    @mock.patch('concurrent.futures.ProcessPoolExecutor')
    def test_compile_pool_called(self, pool_mock):
        if False:
            i = 10
            return i + 15
        compileall.compile_dir(self.directory, quiet=True, workers=5)
        self.assertTrue(pool_mock.called)

    def test_compile_workers_non_positive(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'workers must be greater or equal to 0'):
            compileall.compile_dir(self.directory, workers=-1)

    @skipUnless(_have_multiprocessing, 'requires multiprocessing')
    @mock.patch('concurrent.futures.ProcessPoolExecutor')
    def test_compile_workers_cpu_count(self, pool_mock):
        if False:
            i = 10
            return i + 15
        compileall.compile_dir(self.directory, quiet=True, workers=0)
        self.assertEqual(pool_mock.call_args[1]['max_workers'], None)

    @skipUnless(_have_multiprocessing, 'requires multiprocessing')
    @mock.patch('concurrent.futures.ProcessPoolExecutor')
    @mock.patch('compileall.compile_file')
    def test_compile_one_worker(self, compile_file_mock, pool_mock):
        if False:
            print('Hello World!')
        compileall.compile_dir(self.directory, quiet=True)
        self.assertFalse(pool_mock.called)
        self.assertTrue(compile_file_mock.called)

    @mock.patch('concurrent.futures.ProcessPoolExecutor', new=None)
    @mock.patch('compileall.compile_file')
    def test_compile_missing_multiprocessing(self, compile_file_mock):
        if False:
            while True:
                i = 10
        compileall.compile_dir(self.directory, quiet=True, workers=5)
        self.assertTrue(compile_file_mock.called)

    def test_compile_dir_maxlevels(self):
        if False:
            i = 10
            return i + 15
        depth = 3
        path = self.directory
        for i in range(1, depth + 1):
            path = os.path.join(path, f'dir_{i}')
            source = os.path.join(path, 'script.py')
            os.mkdir(path)
            shutil.copyfile(self.source_path, source)
        pyc_filename = importlib.util.cache_from_source(source)
        compileall.compile_dir(self.directory, quiet=True, maxlevels=depth - 1)
        self.assertFalse(os.path.isfile(pyc_filename))
        compileall.compile_dir(self.directory, quiet=True, maxlevels=depth)
        self.assertTrue(os.path.isfile(pyc_filename))

    def _test_ddir_only(self, *, ddir, parallel=True):
        if False:
            i = 10
            return i + 15
        'Recursive compile_dir ddir must contain package paths; bpo39769.'
        fullpath = ['test', 'foo']
        path = self.directory
        mods = []
        for subdir in fullpath:
            path = os.path.join(path, subdir)
            os.mkdir(path)
            script_helper.make_script(path, '__init__', '')
            mods.append(script_helper.make_script(path, 'mod', 'def fn(): 1/0\nfn()\n'))
        compileall.compile_dir(self.directory, quiet=True, ddir=ddir, workers=2 if parallel else 1)
        self.assertTrue(mods)
        for mod in mods:
            self.assertTrue(mod.startswith(self.directory), mod)
            modcode = importlib.util.cache_from_source(mod)
            modpath = mod[len(self.directory + os.sep):]
            (_, _, err) = script_helper.assert_python_failure(modcode)
            expected_in = os.path.join(ddir, modpath)
            mod_code_obj = test.test_importlib.util.get_code_from_pyc(modcode)
            self.assertEqual(mod_code_obj.co_filename, expected_in)
            self.assertIn(f'"{expected_in}"', os.fsdecode(err))

    def test_ddir_only_one_worker(self):
        if False:
            print('Hello World!')
        'Recursive compile_dir ddir= contains package paths; bpo39769.'
        return self._test_ddir_only(ddir='<a prefix>', parallel=False)

    def test_ddir_multiple_workers(self):
        if False:
            print('Hello World!')
        'Recursive compile_dir ddir= contains package paths; bpo39769.'
        return self._test_ddir_only(ddir='<a prefix>', parallel=True)

    def test_ddir_empty_only_one_worker(self):
        if False:
            i = 10
            return i + 15
        "Recursive compile_dir ddir='' contains package paths; bpo39769."
        return self._test_ddir_only(ddir='', parallel=False)

    def test_ddir_empty_multiple_workers(self):
        if False:
            for i in range(10):
                print('nop')
        "Recursive compile_dir ddir='' contains package paths; bpo39769."
        return self._test_ddir_only(ddir='', parallel=True)

    def test_strip_only(self):
        if False:
            print('Hello World!')
        fullpath = ['test', 'build', 'real', 'path']
        path = os.path.join(self.directory, *fullpath)
        os.makedirs(path)
        script = script_helper.make_script(path, 'test', '1 / 0')
        bc = importlib.util.cache_from_source(script)
        stripdir = os.path.join(self.directory, *fullpath[:2])
        compileall.compile_dir(path, quiet=True, stripdir=stripdir)
        (rc, out, err) = script_helper.assert_python_failure(bc)
        expected_in = os.path.join(*fullpath[2:])
        self.assertIn(expected_in, str(err, encoding=sys.getdefaultencoding()))
        self.assertNotIn(stripdir, str(err, encoding=sys.getdefaultencoding()))

    def test_prepend_only(self):
        if False:
            print('Hello World!')
        fullpath = ['test', 'build', 'real', 'path']
        path = os.path.join(self.directory, *fullpath)
        os.makedirs(path)
        script = script_helper.make_script(path, 'test', '1 / 0')
        bc = importlib.util.cache_from_source(script)
        prependdir = '/foo'
        compileall.compile_dir(path, quiet=True, prependdir=prependdir)
        (rc, out, err) = script_helper.assert_python_failure(bc)
        expected_in = os.path.join(prependdir, self.directory, *fullpath)
        self.assertIn(expected_in, str(err, encoding=sys.getdefaultencoding()))

    def test_strip_and_prepend(self):
        if False:
            for i in range(10):
                print('nop')
        fullpath = ['test', 'build', 'real', 'path']
        path = os.path.join(self.directory, *fullpath)
        os.makedirs(path)
        script = script_helper.make_script(path, 'test', '1 / 0')
        bc = importlib.util.cache_from_source(script)
        stripdir = os.path.join(self.directory, *fullpath[:2])
        prependdir = '/foo'
        compileall.compile_dir(path, quiet=True, stripdir=stripdir, prependdir=prependdir)
        (rc, out, err) = script_helper.assert_python_failure(bc)
        expected_in = os.path.join(prependdir, *fullpath[2:])
        self.assertIn(expected_in, str(err, encoding=sys.getdefaultencoding()))
        self.assertNotIn(stripdir, str(err, encoding=sys.getdefaultencoding()))

    def test_strip_prepend_and_ddir(self):
        if False:
            return 10
        fullpath = ['test', 'build', 'real', 'path', 'ddir']
        path = os.path.join(self.directory, *fullpath)
        os.makedirs(path)
        script_helper.make_script(path, 'test', '1 / 0')
        with self.assertRaises(ValueError):
            compileall.compile_dir(path, quiet=True, ddir='/bar', stripdir='/foo', prependdir='/bar')

    def test_multiple_optimization_levels(self):
        if False:
            return 10
        script = script_helper.make_script(self.directory, 'test_optimization', 'a = 0')
        bc = []
        for opt_level in ('', 1, 2, 3):
            bc.append(importlib.util.cache_from_source(script, optimization=opt_level))
        test_combinations = [[0, 1], [1, 2], [0, 2], [0, 1, 2]]
        for opt_combination in test_combinations:
            compileall.compile_file(script, quiet=True, optimize=opt_combination)
            for opt_level in opt_combination:
                self.assertTrue(os.path.isfile(bc[opt_level]))
                try:
                    os.unlink(bc[opt_level])
                except Exception:
                    pass

    @os_helper.skip_unless_symlink
    def test_ignore_symlink_destination(self):
        if False:
            i = 10
            return i + 15
        allowed_path = os.path.join(self.directory, 'test', 'dir', 'allowed')
        symlinks_path = os.path.join(self.directory, 'test', 'dir', 'symlinks')
        prohibited_path = os.path.join(self.directory, 'test', 'dir', 'prohibited')
        os.makedirs(allowed_path)
        os.makedirs(symlinks_path)
        os.makedirs(prohibited_path)
        allowed_script = script_helper.make_script(allowed_path, 'test_allowed', 'a = 0')
        prohibited_script = script_helper.make_script(prohibited_path, 'test_prohibited', 'a = 0')
        allowed_symlink = os.path.join(symlinks_path, 'test_allowed.py')
        prohibited_symlink = os.path.join(symlinks_path, 'test_prohibited.py')
        os.symlink(allowed_script, allowed_symlink)
        os.symlink(prohibited_script, prohibited_symlink)
        allowed_bc = importlib.util.cache_from_source(allowed_symlink)
        prohibited_bc = importlib.util.cache_from_source(prohibited_symlink)
        compileall.compile_dir(symlinks_path, quiet=True, limit_sl_dest=allowed_path)
        self.assertTrue(os.path.isfile(allowed_bc))
        self.assertFalse(os.path.isfile(prohibited_bc))

class CompileallTestsWithSourceEpoch(CompileallTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=True):
    pass

class CompileallTestsWithoutSourceEpoch(CompileallTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=False):
    pass

class EncodingTest(unittest.TestCase):
    """Issue 6716: compileall should escape source code when printing errors
    to stdout."""

    def setUp(self):
        if False:
            return 10
        self.directory = tempfile.mkdtemp()
        self.source_path = os.path.join(self.directory, '_test.py')
        with open(self.source_path, 'w', encoding='utf-8') as file:
            file.write('# -*- coding: utf-8 -*-\n')
            file.write('print u"€"\n')

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.directory)

    def test_error(self):
        if False:
            return 10
        try:
            orig_stdout = sys.stdout
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='ascii')
            compileall.compile_dir(self.directory)
        finally:
            sys.stdout = orig_stdout

class CommandLineTestsBase:
    """Test compileall's CLI."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.directory = tempfile.mkdtemp()
        self.addCleanup(os_helper.rmtree, self.directory)
        self.pkgdir = os.path.join(self.directory, 'foo')
        os.mkdir(self.pkgdir)
        self.pkgdir_cachedir = os.path.join(self.pkgdir, '__pycache__')
        self.initfn = script_helper.make_script(self.pkgdir, '__init__', '')
        self.barfn = script_helper.make_script(self.pkgdir, 'bar', '')

    @contextlib.contextmanager
    def temporary_pycache_prefix(self):
        if False:
            print('Hello World!')
        'Adjust and restore sys.pycache_prefix.'
        old_prefix = sys.pycache_prefix
        new_prefix = os.path.join(self.directory, '__testcache__')
        try:
            sys.pycache_prefix = new_prefix
            yield {'PYTHONPATH': self.directory, 'PYTHONPYCACHEPREFIX': new_prefix}
        finally:
            sys.pycache_prefix = old_prefix

    def _get_run_args(self, args):
        if False:
            print('Hello World!')
        return [*support.optim_args_from_interpreter_flags(), '-S', '-m', 'compileall', *args]

    def assertRunOK(self, *args, **env_vars):
        if False:
            i = 10
            return i + 15
        (rc, out, err) = script_helper.assert_python_ok(*self._get_run_args(args), **env_vars, PYTHONIOENCODING='utf-8')
        self.assertEqual(b'', err)
        return out

    def assertRunNotOK(self, *args, **env_vars):
        if False:
            print('Hello World!')
        (rc, out, err) = script_helper.assert_python_failure(*self._get_run_args(args), **env_vars, PYTHONIOENCODING='utf-8')
        return (rc, out, err)

    def assertCompiled(self, fn):
        if False:
            return 10
        path = importlib.util.cache_from_source(fn)
        self.assertTrue(os.path.exists(path))

    def assertNotCompiled(self, fn):
        if False:
            i = 10
            return i + 15
        path = importlib.util.cache_from_source(fn)
        self.assertFalse(os.path.exists(path))

    def test_no_args_compiles_path(self):
        if False:
            for i in range(10):
                print('nop')
        bazfn = script_helper.make_script(self.directory, 'baz', '')
        with self.temporary_pycache_prefix() as env:
            self.assertRunOK(**env)
            self.assertCompiled(bazfn)
            self.assertNotCompiled(self.initfn)
            self.assertNotCompiled(self.barfn)

    @without_source_date_epoch
    def test_no_args_respects_force_flag(self):
        if False:
            return 10
        bazfn = script_helper.make_script(self.directory, 'baz', '')
        with self.temporary_pycache_prefix() as env:
            self.assertRunOK(**env)
            pycpath = importlib.util.cache_from_source(bazfn)
        os.utime(pycpath, (time.time() - 60,) * 2)
        mtime = os.stat(pycpath).st_mtime
        self.assertRunOK(**env)
        mtime2 = os.stat(pycpath).st_mtime
        self.assertEqual(mtime, mtime2)
        self.assertRunOK('-f', **env)
        mtime2 = os.stat(pycpath).st_mtime
        self.assertNotEqual(mtime, mtime2)

    def test_no_args_respects_quiet_flag(self):
        if False:
            while True:
                i = 10
        script_helper.make_script(self.directory, 'baz', '')
        with self.temporary_pycache_prefix() as env:
            noisy = self.assertRunOK(**env)
        self.assertIn(b'Listing ', noisy)
        quiet = self.assertRunOK('-q', **env)
        self.assertNotIn(b'Listing ', quiet)
    for (name, ext, switch) in [('normal', 'pyc', []), ('optimize', 'opt-1.pyc', ['-O']), ('doubleoptimize', 'opt-2.pyc', ['-OO'])]:

        def f(self, ext=ext, switch=switch):
            if False:
                return 10
            script_helper.assert_python_ok(*switch + ['-m', 'compileall', '-q', self.pkgdir])
            self.assertTrue(os.path.exists(self.pkgdir_cachedir))
            expected = sorted((base.format(sys.implementation.cache_tag, ext) for base in ('__init__.{}.{}', 'bar.{}.{}')))
            self.assertEqual(sorted(os.listdir(self.pkgdir_cachedir)), expected)
            self.assertFalse([fn for fn in os.listdir(self.pkgdir) if fn.endswith(ext)])
        locals()['test_pep3147_paths_' + name] = f

    def test_legacy_paths(self):
        if False:
            while True:
                i = 10
        self.assertRunOK('-b', '-q', self.pkgdir)
        self.assertFalse(os.path.exists(self.pkgdir_cachedir))
        expected = sorted(['__init__.py', '__init__.pyc', 'bar.py', 'bar.pyc'])
        self.assertEqual(sorted(os.listdir(self.pkgdir)), expected)

    def test_multiple_runs(self):
        if False:
            i = 10
            return i + 15
        self.assertRunOK('-q', self.pkgdir)
        self.assertTrue(os.path.exists(self.pkgdir_cachedir))
        cachecachedir = os.path.join(self.pkgdir_cachedir, '__pycache__')
        self.assertFalse(os.path.exists(cachecachedir))
        self.assertRunOK('-q', self.pkgdir)
        self.assertTrue(os.path.exists(self.pkgdir_cachedir))
        self.assertFalse(os.path.exists(cachecachedir))

    @without_source_date_epoch
    def test_force(self):
        if False:
            while True:
                i = 10
        self.assertRunOK('-q', self.pkgdir)
        pycpath = importlib.util.cache_from_source(self.barfn)
        os.utime(pycpath, (time.time() - 60,) * 2)
        mtime = os.stat(pycpath).st_mtime
        self.assertRunOK('-q', self.pkgdir)
        mtime2 = os.stat(pycpath).st_mtime
        self.assertEqual(mtime, mtime2)
        self.assertRunOK('-q', '-f', self.pkgdir)
        mtime2 = os.stat(pycpath).st_mtime
        self.assertNotEqual(mtime, mtime2)

    def test_recursion_control(self):
        if False:
            print('Hello World!')
        subpackage = os.path.join(self.pkgdir, 'spam')
        os.mkdir(subpackage)
        subinitfn = script_helper.make_script(subpackage, '__init__', '')
        hamfn = script_helper.make_script(subpackage, 'ham', '')
        self.assertRunOK('-q', '-l', self.pkgdir)
        self.assertNotCompiled(subinitfn)
        self.assertFalse(os.path.exists(os.path.join(subpackage, '__pycache__')))
        self.assertRunOK('-q', self.pkgdir)
        self.assertCompiled(subinitfn)
        self.assertCompiled(hamfn)

    def test_recursion_limit(self):
        if False:
            return 10
        subpackage = os.path.join(self.pkgdir, 'spam')
        subpackage2 = os.path.join(subpackage, 'ham')
        subpackage3 = os.path.join(subpackage2, 'eggs')
        for pkg in (subpackage, subpackage2, subpackage3):
            script_helper.make_pkg(pkg)
        subinitfn = os.path.join(subpackage, '__init__.py')
        hamfn = script_helper.make_script(subpackage, 'ham', '')
        spamfn = script_helper.make_script(subpackage2, 'spam', '')
        eggfn = script_helper.make_script(subpackage3, 'egg', '')
        self.assertRunOK('-q', '-r 0', self.pkgdir)
        self.assertNotCompiled(subinitfn)
        self.assertFalse(os.path.exists(os.path.join(subpackage, '__pycache__')))
        self.assertRunOK('-q', '-r 1', self.pkgdir)
        self.assertCompiled(subinitfn)
        self.assertCompiled(hamfn)
        self.assertNotCompiled(spamfn)
        self.assertRunOK('-q', '-r 2', self.pkgdir)
        self.assertCompiled(subinitfn)
        self.assertCompiled(hamfn)
        self.assertCompiled(spamfn)
        self.assertNotCompiled(eggfn)
        self.assertRunOK('-q', '-r 5', self.pkgdir)
        self.assertCompiled(subinitfn)
        self.assertCompiled(hamfn)
        self.assertCompiled(spamfn)
        self.assertCompiled(eggfn)

    @os_helper.skip_unless_symlink
    def test_symlink_loop(self):
        if False:
            return 10
        pkg = os.path.join(self.pkgdir, 'spam')
        script_helper.make_pkg(pkg)
        os.symlink('.', os.path.join(pkg, 'evil'))
        os.symlink('.', os.path.join(pkg, 'evil2'))
        self.assertRunOK('-q', self.pkgdir)
        self.assertCompiled(os.path.join(self.pkgdir, 'spam', 'evil', 'evil2', '__init__.py'))

    def test_quiet(self):
        if False:
            while True:
                i = 10
        noisy = self.assertRunOK(self.pkgdir)
        quiet = self.assertRunOK('-q', self.pkgdir)
        self.assertNotEqual(b'', noisy)
        self.assertEqual(b'', quiet)

    def test_silent(self):
        if False:
            while True:
                i = 10
        script_helper.make_script(self.pkgdir, 'crunchyfrog', 'bad(syntax')
        (_, quiet, _) = self.assertRunNotOK('-q', self.pkgdir)
        (_, silent, _) = self.assertRunNotOK('-qq', self.pkgdir)
        self.assertNotEqual(b'', quiet)
        self.assertEqual(b'', silent)

    def test_regexp(self):
        if False:
            print('Hello World!')
        self.assertRunOK('-q', '-x', 'ba[^\\\\/]*$', self.pkgdir)
        self.assertNotCompiled(self.barfn)
        self.assertCompiled(self.initfn)

    def test_multiple_dirs(self):
        if False:
            return 10
        pkgdir2 = os.path.join(self.directory, 'foo2')
        os.mkdir(pkgdir2)
        init2fn = script_helper.make_script(pkgdir2, '__init__', '')
        bar2fn = script_helper.make_script(pkgdir2, 'bar2', '')
        self.assertRunOK('-q', self.pkgdir, pkgdir2)
        self.assertCompiled(self.initfn)
        self.assertCompiled(self.barfn)
        self.assertCompiled(init2fn)
        self.assertCompiled(bar2fn)

    def test_d_compile_error(self):
        if False:
            i = 10
            return i + 15
        script_helper.make_script(self.pkgdir, 'crunchyfrog', 'bad(syntax')
        (rc, out, err) = self.assertRunNotOK('-q', '-d', 'dinsdale', self.pkgdir)
        self.assertRegex(out, b'File "dinsdale')

    def test_d_runtime_error(self):
        if False:
            for i in range(10):
                print('nop')
        bazfn = script_helper.make_script(self.pkgdir, 'baz', 'raise Exception')
        self.assertRunOK('-q', '-d', 'dinsdale', self.pkgdir)
        fn = script_helper.make_script(self.pkgdir, 'bing', 'import baz')
        pyc = importlib.util.cache_from_source(bazfn)
        os.rename(pyc, os.path.join(self.pkgdir, 'baz.pyc'))
        os.remove(bazfn)
        (rc, out, err) = script_helper.assert_python_failure(fn, __isolated=False)
        self.assertRegex(err, b'File "dinsdale')

    def test_include_bad_file(self):
        if False:
            return 10
        (rc, out, err) = self.assertRunNotOK('-i', os.path.join(self.directory, 'nosuchfile'), self.pkgdir)
        self.assertRegex(out, b'rror.*nosuchfile')
        self.assertNotRegex(err, b'Traceback')
        self.assertFalse(os.path.exists(importlib.util.cache_from_source(self.pkgdir_cachedir)))

    def test_include_file_with_arg(self):
        if False:
            return 10
        f1 = script_helper.make_script(self.pkgdir, 'f1', '')
        f2 = script_helper.make_script(self.pkgdir, 'f2', '')
        f3 = script_helper.make_script(self.pkgdir, 'f3', '')
        f4 = script_helper.make_script(self.pkgdir, 'f4', '')
        with open(os.path.join(self.directory, 'l1'), 'w', encoding='utf-8') as l1:
            l1.write(os.path.join(self.pkgdir, 'f1.py') + os.linesep)
            l1.write(os.path.join(self.pkgdir, 'f2.py') + os.linesep)
        self.assertRunOK('-i', os.path.join(self.directory, 'l1'), f4)
        self.assertCompiled(f1)
        self.assertCompiled(f2)
        self.assertNotCompiled(f3)
        self.assertCompiled(f4)

    def test_include_file_no_arg(self):
        if False:
            print('Hello World!')
        f1 = script_helper.make_script(self.pkgdir, 'f1', '')
        f2 = script_helper.make_script(self.pkgdir, 'f2', '')
        f3 = script_helper.make_script(self.pkgdir, 'f3', '')
        f4 = script_helper.make_script(self.pkgdir, 'f4', '')
        with open(os.path.join(self.directory, 'l1'), 'w', encoding='utf-8') as l1:
            l1.write(os.path.join(self.pkgdir, 'f2.py') + os.linesep)
        self.assertRunOK('-i', os.path.join(self.directory, 'l1'))
        self.assertNotCompiled(f1)
        self.assertCompiled(f2)
        self.assertNotCompiled(f3)
        self.assertNotCompiled(f4)

    def test_include_on_stdin(self):
        if False:
            return 10
        f1 = script_helper.make_script(self.pkgdir, 'f1', '')
        f2 = script_helper.make_script(self.pkgdir, 'f2', '')
        f3 = script_helper.make_script(self.pkgdir, 'f3', '')
        f4 = script_helper.make_script(self.pkgdir, 'f4', '')
        p = script_helper.spawn_python(*self._get_run_args(()) + ['-i', '-'])
        p.stdin.write((f3 + os.linesep).encode('ascii'))
        script_helper.kill_python(p)
        self.assertNotCompiled(f1)
        self.assertNotCompiled(f2)
        self.assertCompiled(f3)
        self.assertNotCompiled(f4)

    def test_compiles_as_much_as_possible(self):
        if False:
            while True:
                i = 10
        bingfn = script_helper.make_script(self.pkgdir, 'bing', 'syntax(error')
        (rc, out, err) = self.assertRunNotOK('nosuchfile', self.initfn, bingfn, self.barfn)
        self.assertRegex(out, b'rror')
        self.assertNotCompiled(bingfn)
        self.assertCompiled(self.initfn)
        self.assertCompiled(self.barfn)

    def test_invalid_arg_produces_message(self):
        if False:
            while True:
                i = 10
        out = self.assertRunOK('badfilename')
        self.assertRegex(out, b"Can't list 'badfilename'")

    def test_pyc_invalidation_mode(self):
        if False:
            return 10
        script_helper.make_script(self.pkgdir, 'f1', '')
        pyc = importlib.util.cache_from_source(os.path.join(self.pkgdir, 'f1.py'))
        self.assertRunOK('--invalidation-mode=checked-hash', self.pkgdir)
        with open(pyc, 'rb') as fp:
            data = fp.read()
        self.assertEqual(int.from_bytes(data[4:8], 'little'), 3)
        self.assertRunOK('--invalidation-mode=unchecked-hash', self.pkgdir)
        with open(pyc, 'rb') as fp:
            data = fp.read()
        self.assertEqual(int.from_bytes(data[4:8], 'little'), 1)

    @skipUnless(_have_multiprocessing, 'requires multiprocessing')
    def test_workers(self):
        if False:
            return 10
        bar2fn = script_helper.make_script(self.directory, 'bar2', '')
        files = []
        for suffix in range(5):
            pkgdir = os.path.join(self.directory, 'foo{}'.format(suffix))
            os.mkdir(pkgdir)
            fn = script_helper.make_script(pkgdir, '__init__', '')
            files.append(script_helper.make_script(pkgdir, 'bar2', ''))
        self.assertRunOK(self.directory, '-j', '0')
        self.assertCompiled(bar2fn)
        for file in files:
            self.assertCompiled(file)

    @mock.patch('compileall.compile_dir')
    def test_workers_available_cores(self, compile_dir):
        if False:
            print('Hello World!')
        with mock.patch('sys.argv', new=[sys.executable, self.directory, '-j0']):
            compileall.main()
            self.assertTrue(compile_dir.called)
            self.assertEqual(compile_dir.call_args[-1]['workers'], 0)

    def test_strip_and_prepend(self):
        if False:
            while True:
                i = 10
        fullpath = ['test', 'build', 'real', 'path']
        path = os.path.join(self.directory, *fullpath)
        os.makedirs(path)
        script = script_helper.make_script(path, 'test', '1 / 0')
        bc = importlib.util.cache_from_source(script)
        stripdir = os.path.join(self.directory, *fullpath[:2])
        prependdir = '/foo'
        self.assertRunOK('-s', stripdir, '-p', prependdir, path)
        (rc, out, err) = script_helper.assert_python_failure(bc)
        expected_in = os.path.join(prependdir, *fullpath[2:])
        self.assertIn(expected_in, str(err, encoding=sys.getdefaultencoding()))
        self.assertNotIn(stripdir, str(err, encoding=sys.getdefaultencoding()))

    def test_multiple_optimization_levels(self):
        if False:
            return 10
        path = os.path.join(self.directory, 'optimizations')
        os.makedirs(path)
        script = script_helper.make_script(path, 'test_optimization', 'a = 0')
        bc = []
        for opt_level in ('', 1, 2, 3):
            bc.append(importlib.util.cache_from_source(script, optimization=opt_level))
        test_combinations = [['0', '1'], ['1', '2'], ['0', '2'], ['0', '1', '2']]
        for opt_combination in test_combinations:
            self.assertRunOK(path, *('-o' + str(n) for n in opt_combination))
            for opt_level in opt_combination:
                self.assertTrue(os.path.isfile(bc[int(opt_level)]))
                try:
                    os.unlink(bc[opt_level])
                except Exception:
                    pass

    @os_helper.skip_unless_symlink
    def test_ignore_symlink_destination(self):
        if False:
            i = 10
            return i + 15
        allowed_path = os.path.join(self.directory, 'test', 'dir', 'allowed')
        symlinks_path = os.path.join(self.directory, 'test', 'dir', 'symlinks')
        prohibited_path = os.path.join(self.directory, 'test', 'dir', 'prohibited')
        os.makedirs(allowed_path)
        os.makedirs(symlinks_path)
        os.makedirs(prohibited_path)
        allowed_script = script_helper.make_script(allowed_path, 'test_allowed', 'a = 0')
        prohibited_script = script_helper.make_script(prohibited_path, 'test_prohibited', 'a = 0')
        allowed_symlink = os.path.join(symlinks_path, 'test_allowed.py')
        prohibited_symlink = os.path.join(symlinks_path, 'test_prohibited.py')
        os.symlink(allowed_script, allowed_symlink)
        os.symlink(prohibited_script, prohibited_symlink)
        allowed_bc = importlib.util.cache_from_source(allowed_symlink)
        prohibited_bc = importlib.util.cache_from_source(prohibited_symlink)
        self.assertRunOK(symlinks_path, '-e', allowed_path)
        self.assertTrue(os.path.isfile(allowed_bc))
        self.assertFalse(os.path.isfile(prohibited_bc))

    def test_hardlink_bad_args(self):
        if False:
            return 10
        self.assertRunNotOK(self.directory, '-o 1', '--hardlink-dupes')

    def test_hardlink(self):
        if False:
            print('Hello World!')
        for dedup in (True, False):
            with tempfile.TemporaryDirectory() as path:
                with self.subTest(dedup=dedup):
                    script = script_helper.make_script(path, 'script', 'a = 0')
                    pycs = get_pycs(script)
                    args = ['-q', '-o 0', '-o 1', '-o 2']
                    if dedup:
                        args.append('--hardlink-dupes')
                    self.assertRunOK(path, *args)
                    self.assertEqual(is_hardlink(pycs[0], pycs[1]), dedup)
                    self.assertEqual(is_hardlink(pycs[1], pycs[2]), dedup)
                    self.assertEqual(is_hardlink(pycs[0], pycs[2]), dedup)

class CommandLineTestsWithSourceEpoch(CommandLineTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=True):
    pass

class CommandLineTestsNoSourceEpoch(CommandLineTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=False):
    pass

class HardlinkDedupTestsBase:

    def setUp(self):
        if False:
            print('Hello World!')
        self.path = None

    @contextlib.contextmanager
    def temporary_directory(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as path:
            self.path = path
            yield path
            self.path = None

    def make_script(self, code, name='script'):
        if False:
            for i in range(10):
                print('nop')
        return script_helper.make_script(self.path, name, code)

    def compile_dir(self, *, dedup=True, optimize=(0, 1, 2), force=False):
        if False:
            print('Hello World!')
        compileall.compile_dir(self.path, quiet=True, optimize=optimize, hardlink_dupes=dedup, force=force)

    def test_bad_args(self):
        if False:
            i = 10
            return i + 15
        with self.temporary_directory():
            self.make_script('pass')
            with self.assertRaises(ValueError):
                compileall.compile_dir(self.path, quiet=True, optimize=0, hardlink_dupes=True)
            with self.assertRaises(ValueError):
                compileall.compile_dir(self.path, quiet=True, optimize=[0, 0], hardlink_dupes=True)

    def create_code(self, docstring=False, assertion=False):
        if False:
            return 10
        lines = []
        if docstring:
            lines.append("'module docstring'")
        lines.append('x = 1')
        if assertion:
            lines.append('assert x == 1')
        return '\n'.join(lines)

    def iter_codes(self):
        if False:
            for i in range(10):
                print('nop')
        for docstring in (False, True):
            for assertion in (False, True):
                code = self.create_code(docstring=docstring, assertion=assertion)
                yield (code, docstring, assertion)

    def test_disabled(self):
        if False:
            print('Hello World!')
        for (code, docstring, assertion) in self.iter_codes():
            with self.subTest(docstring=docstring, assertion=assertion):
                with self.temporary_directory():
                    script = self.make_script(code)
                    pycs = get_pycs(script)
                    self.compile_dir(dedup=False)
                    self.assertFalse(is_hardlink(pycs[0], pycs[1]))
                    self.assertFalse(is_hardlink(pycs[0], pycs[2]))
                    self.assertFalse(is_hardlink(pycs[1], pycs[2]))

    def check_hardlinks(self, script, docstring=False, assertion=False):
        if False:
            print('Hello World!')
        pycs = get_pycs(script)
        self.assertEqual(is_hardlink(pycs[0], pycs[1]), not assertion)
        self.assertEqual(is_hardlink(pycs[0], pycs[2]), not assertion and (not docstring))
        self.assertEqual(is_hardlink(pycs[1], pycs[2]), not docstring)

    def test_hardlink(self):
        if False:
            i = 10
            return i + 15
        for (code, docstring, assertion) in self.iter_codes():
            with self.subTest(docstring=docstring, assertion=assertion):
                with self.temporary_directory():
                    script = self.make_script(code)
                    self.compile_dir()
                    self.check_hardlinks(script, docstring, assertion)

    def test_only_two_levels(self):
        if False:
            for i in range(10):
                print('nop')
        for opts in ((0, 1), (1, 2), (0, 2)):
            with self.subTest(opts=opts):
                with self.temporary_directory():
                    script = self.make_script(self.create_code())
                    self.compile_dir(optimize=opts)
                    pyc1 = get_pyc(script, opts[0])
                    pyc2 = get_pyc(script, opts[1])
                    self.assertTrue(is_hardlink(pyc1, pyc2))

    def test_duplicated_levels(self):
        if False:
            i = 10
            return i + 15
        with self.temporary_directory():
            script = self.make_script(self.create_code())
            self.compile_dir(optimize=[1, 0, 1, 0])
            pyc1 = get_pyc(script, 0)
            pyc2 = get_pyc(script, 1)
            self.assertTrue(is_hardlink(pyc1, pyc2))

    def test_recompilation(self):
        if False:
            for i in range(10):
                print('nop')
        with self.temporary_directory():
            script = self.make_script('a = 0')
            self.compile_dir()
            self.check_hardlinks(script)
            pycs = get_pycs(script)
            inode = os.stat(pycs[0]).st_ino
            script = self.make_script('print(0)')
            self.compile_dir(optimize=[0, 2], force=True)
            self.assertEqual(inode, os.stat(pycs[1]).st_ino)
            self.assertTrue(is_hardlink(pycs[0], pycs[2]))
            self.assertNotEqual(inode, os.stat(pycs[2]).st_ino)
            self.assertFalse(filecmp.cmp(pycs[1], pycs[2], shallow=True))

    def test_import(self):
        if False:
            while True:
                i = 10
        with self.temporary_directory():
            script = self.make_script(self.create_code(), name='module')
            self.compile_dir()
            self.check_hardlinks(script)
            pycs = get_pycs(script)
            inode = os.stat(pycs[0]).st_ino
            script = self.make_script('print(0)', name='module')
            script_helper.assert_python_ok('-O', '-c', 'import module', __isolated=False, PYTHONPATH=self.path)
            self.assertEqual(inode, os.stat(pycs[0]).st_ino)
            self.assertEqual(inode, os.stat(pycs[2]).st_ino)
            self.assertFalse(is_hardlink(pycs[1], pycs[2]))
            self.assertFalse(filecmp.cmp(pycs[1], pycs[2], shallow=True))

class HardlinkDedupTestsWithSourceEpoch(HardlinkDedupTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=True):
    pass

class HardlinkDedupTestsNoSourceEpoch(HardlinkDedupTestsBase, unittest.TestCase, metaclass=SourceDateEpochTestMeta, source_date_epoch=False):
    pass
if __name__ == '__main__':
    unittest.main()