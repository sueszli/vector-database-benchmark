"""Tests for yapf.file_resources."""
import codecs
import contextlib
import os
import shutil
import tempfile
import unittest
from io import BytesIO
from yapf.yapflib import errors
from yapf.yapflib import file_resources
from yapftests import utils
from yapftests import yapf_test_helper

@contextlib.contextmanager
def _restore_working_dir():
    if False:
        for i in range(10):
            print('nop')
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

@contextlib.contextmanager
def _exists_mocked_in_module(module, mock_implementation):
    if False:
        i = 10
        return i + 15
    unmocked_exists = getattr(module, 'exists')
    setattr(module, 'exists', mock_implementation)
    try:
        yield
    finally:
        setattr(module, 'exists', unmocked_exists)

class GetExcludePatternsForDir(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            return 10
        self.test_tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.test_tmpdir)

    def test_get_exclude_file_patterns_from_yapfignore(self):
        if False:
            for i in range(10):
                print('nop')
        local_ignore_file = os.path.join(self.test_tmpdir, '.yapfignore')
        ignore_patterns = ['temp/**/*.py', 'temp2/*.py']
        with open(local_ignore_file, 'w') as f:
            f.writelines('\n'.join(ignore_patterns))
        self.assertEqual(sorted(file_resources.GetExcludePatternsForDir(self.test_tmpdir)), sorted(ignore_patterns))

    def test_get_exclude_file_patterns_from_yapfignore_with_wrong_syntax(self):
        if False:
            while True:
                i = 10
        local_ignore_file = os.path.join(self.test_tmpdir, '.yapfignore')
        ignore_patterns = ['temp/**/*.py', './wrong/syntax/*.py']
        with open(local_ignore_file, 'w') as f:
            f.writelines('\n'.join(ignore_patterns))
        with self.assertRaises(errors.YapfError):
            file_resources.GetExcludePatternsForDir(self.test_tmpdir)

    def test_get_exclude_file_patterns_from_pyproject(self):
        if False:
            while True:
                i = 10
        local_ignore_file = os.path.join(self.test_tmpdir, 'pyproject.toml')
        ignore_patterns = ['temp/**/*.py', 'temp2/*.py']
        with open(local_ignore_file, 'w') as f:
            f.write('[tool.yapfignore]\n')
            f.write('ignore_patterns=[')
            f.writelines('\n,'.join(['"{}"'.format(p) for p in ignore_patterns]))
            f.write(']')
        self.assertEqual(sorted(file_resources.GetExcludePatternsForDir(self.test_tmpdir)), sorted(ignore_patterns))

    def test_get_exclude_file_patterns_from_pyproject_no_ignore_section(self):
        if False:
            for i in range(10):
                print('nop')
        local_ignore_file = os.path.join(self.test_tmpdir, 'pyproject.toml')
        ignore_patterns = []
        open(local_ignore_file, 'w').close()
        self.assertEqual(sorted(file_resources.GetExcludePatternsForDir(self.test_tmpdir)), sorted(ignore_patterns))

    def test_get_exclude_file_patterns_from_pyproject_ignore_section_empty(self):
        if False:
            i = 10
            return i + 15
        local_ignore_file = os.path.join(self.test_tmpdir, 'pyproject.toml')
        ignore_patterns = []
        with open(local_ignore_file, 'w') as f:
            f.write('[tool.yapfignore]\n')
        self.assertEqual(sorted(file_resources.GetExcludePatternsForDir(self.test_tmpdir)), sorted(ignore_patterns))

    def test_get_exclude_file_patterns_with_no_config_files(self):
        if False:
            return 10
        ignore_patterns = []
        self.assertEqual(sorted(file_resources.GetExcludePatternsForDir(self.test_tmpdir)), sorted(ignore_patterns))

class GetDefaultStyleForDirTest(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.test_tmpdir)

    def test_no_local_style(self):
        if False:
            print('Hello World!')
        test_file = os.path.join(self.test_tmpdir, 'file.py')
        style_name = file_resources.GetDefaultStyleForDir(test_file)
        self.assertEqual(style_name, 'pep8')

    def test_no_local_style_custom_default(self):
        if False:
            i = 10
            return i + 15
        test_file = os.path.join(self.test_tmpdir, 'file.py')
        style_name = file_resources.GetDefaultStyleForDir(test_file, default_style='custom-default')
        self.assertEqual(style_name, 'custom-default')

    def test_with_local_style(self):
        if False:
            for i in range(10):
                print('nop')
        style_file = os.path.join(self.test_tmpdir, '.style.yapf')
        open(style_file, 'w').close()
        test_filename = os.path.join(self.test_tmpdir, 'file.py')
        self.assertEqual(style_file, file_resources.GetDefaultStyleForDir(test_filename))
        test_filename = os.path.join(self.test_tmpdir, 'dir1', 'file.py')
        self.assertEqual(style_file, file_resources.GetDefaultStyleForDir(test_filename))

    def test_setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        setup_config = os.path.join(self.test_tmpdir, 'setup.cfg')
        open(setup_config, 'w').close()
        test_dir = os.path.join(self.test_tmpdir, 'dir1')
        style_name = file_resources.GetDefaultStyleForDir(test_dir)
        self.assertEqual(style_name, 'pep8')
        with open(setup_config, 'w') as f:
            f.write('[yapf]\n')
        self.assertEqual(setup_config, file_resources.GetDefaultStyleForDir(test_dir))

    def test_pyproject_toml(self):
        if False:
            while True:
                i = 10
        pyproject_toml = os.path.join(self.test_tmpdir, 'pyproject.toml')
        open(pyproject_toml, 'w').close()
        test_dir = os.path.join(self.test_tmpdir, 'dir1')
        style_name = file_resources.GetDefaultStyleForDir(test_dir)
        self.assertEqual(style_name, 'pep8')
        with open(pyproject_toml, 'w') as f:
            f.write('[tool.yapf]\n')
        self.assertEqual(pyproject_toml, file_resources.GetDefaultStyleForDir(test_dir))

    def test_local_style_at_root(self):
        if False:
            print('Hello World!')
        rootdir = os.path.abspath(os.path.sep)
        test_dir_at_root = os.path.join(rootdir, 'dir1')
        test_dir_under_root = os.path.join(rootdir, 'dir1', 'dir2')
        style_file = os.path.join(rootdir, '.style.yapf')

        def mock_exists_implementation(path):
            if False:
                while True:
                    i = 10
            return path == style_file
        with _exists_mocked_in_module(file_resources.os.path, mock_exists_implementation):
            default_style_at_root = file_resources.GetDefaultStyleForDir(test_dir_at_root)
            self.assertEqual(style_file, default_style_at_root)
            default_style_under_root = file_resources.GetDefaultStyleForDir(test_dir_under_root)
            self.assertEqual(style_file, default_style_under_root)

def _touch_files(filenames):
    if False:
        print('Hello World!')
    for name in filenames:
        open(name, 'a').close()

class GetCommandLineFilesTest(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_tmpdir = tempfile.mkdtemp()
        self.old_dir = os.getcwd()

    def tearDown(self):
        if False:
            return 10
        os.chdir(self.old_dir)
        shutil.rmtree(self.test_tmpdir)

    def _make_test_dir(self, name):
        if False:
            i = 10
            return i + 15
        fullpath = os.path.normpath(os.path.join(self.test_tmpdir, name))
        os.makedirs(fullpath)
        return fullpath

    def test_find_files_not_dirs(self):
        if False:
            i = 10
            return i + 15
        tdir1 = self._make_test_dir('test1')
        tdir2 = self._make_test_dir('test2')
        file1 = os.path.join(tdir1, 'testfile1.py')
        file2 = os.path.join(tdir2, 'testfile2.py')
        _touch_files([file1, file2])
        self.assertEqual(file_resources.GetCommandLineFiles([file1, file2], recursive=False, exclude=None), [file1, file2])
        self.assertEqual(file_resources.GetCommandLineFiles([file1, file2], recursive=True, exclude=None), [file1, file2])

    def test_nonrecursive_find_in_dir(self):
        if False:
            print('Hello World!')
        tdir1 = self._make_test_dir('test1')
        tdir2 = self._make_test_dir('test1/foo')
        file1 = os.path.join(tdir1, 'testfile1.py')
        file2 = os.path.join(tdir2, 'testfile2.py')
        _touch_files([file1, file2])
        self.assertRaises(errors.YapfError, file_resources.GetCommandLineFiles, command_line_file_list=[tdir1], recursive=False, exclude=None)

    def test_recursive_find_in_dir(self):
        if False:
            for i in range(10):
                print('nop')
        tdir1 = self._make_test_dir('test1')
        tdir2 = self._make_test_dir('test2/testinner/')
        tdir3 = self._make_test_dir('test3/foo/bar/bas/xxx')
        files = [os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]
        _touch_files(files)
        self.assertEqual(sorted(file_resources.GetCommandLineFiles([self.test_tmpdir], recursive=True, exclude=None)), sorted(files))

    def test_recursive_find_in_dir_with_exclude(self):
        if False:
            print('Hello World!')
        tdir1 = self._make_test_dir('test1')
        tdir2 = self._make_test_dir('test2/testinner/')
        tdir3 = self._make_test_dir('test3/foo/bar/bas/xxx')
        files = [os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]
        _touch_files(files)
        self.assertEqual(sorted(file_resources.GetCommandLineFiles([self.test_tmpdir], recursive=True, exclude=['*test*3.py'])), sorted([os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py')]))

    def test_find_with_excluded_hidden_dirs(self):
        if False:
            while True:
                i = 10
        tdir1 = self._make_test_dir('.test1')
        tdir2 = self._make_test_dir('test_2')
        tdir3 = self._make_test_dir('test.3')
        files = [os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]
        _touch_files(files)
        actual = file_resources.GetCommandLineFiles([self.test_tmpdir], recursive=True, exclude=['*.test1*'])
        self.assertEqual(sorted(actual), sorted([os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]))

    def test_find_with_excluded_hidden_dirs_relative(self):
        if False:
            i = 10
            return i + 15
        'Test find with excluded hidden dirs.\n\n    A regression test against a specific case where a hidden directory (one\n    beginning with a period) is being excluded, but it is also an immediate\n    child of the current directory which has been specified in a relative\n    manner.\n\n    At its core, the bug has to do with overzealous stripping of "./foo" so that\n    it removes too much from "./.foo" .\n    '
        tdir1 = self._make_test_dir('.test1')
        tdir2 = self._make_test_dir('test_2')
        tdir3 = self._make_test_dir('test.3')
        files = [os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]
        _touch_files(files)
        with _restore_working_dir():
            os.chdir(self.test_tmpdir)
            actual = file_resources.GetCommandLineFiles([os.path.relpath(self.test_tmpdir)], recursive=True, exclude=['*.test1*'])
            self.assertEqual(sorted(actual), sorted([os.path.join(os.path.relpath(self.test_tmpdir), os.path.basename(tdir2), 'testfile2.py'), os.path.join(os.path.relpath(self.test_tmpdir), os.path.basename(tdir3), 'testfile3.py')]))

    def test_find_with_excluded_dirs(self):
        if False:
            return 10
        tdir1 = self._make_test_dir('test1')
        tdir2 = self._make_test_dir('test2/testinner/')
        tdir3 = self._make_test_dir('test3/foo/bar/bas/xxx')
        files = [os.path.join(tdir1, 'testfile1.py'), os.path.join(tdir2, 'testfile2.py'), os.path.join(tdir3, 'testfile3.py')]
        _touch_files(files)
        os.chdir(self.test_tmpdir)
        found = sorted(file_resources.GetCommandLineFiles(['test1', 'test2', 'test3'], recursive=True, exclude=['test1', 'test2/testinner/']))
        self.assertEqual(found, ['test3/foo/bar/bas/xxx/testfile3.py'.replace('/', os.path.sep)])
        found = sorted(file_resources.GetCommandLineFiles(['.'], recursive=True, exclude=['test1', 'test3']))
        self.assertEqual(found, ['./test2/testinner/testfile2.py'.replace('/', os.path.sep)])

    def test_find_with_excluded_current_dir(self):
        if False:
            print('Hello World!')
        with self.assertRaises(errors.YapfError):
            file_resources.GetCommandLineFiles([], False, exclude=['./z'])

class IsPythonFileTest(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.test_tmpdir)

    def test_with_py_extension(self):
        if False:
            i = 10
            return i + 15
        file1 = os.path.join(self.test_tmpdir, 'testfile1.py')
        self.assertTrue(file_resources.IsPythonFile(file1))

    def test_empty_without_py_extension(self):
        if False:
            return 10
        file1 = os.path.join(self.test_tmpdir, 'testfile1')
        self.assertFalse(file_resources.IsPythonFile(file1))
        file2 = os.path.join(self.test_tmpdir, 'testfile1.rb')
        self.assertFalse(file_resources.IsPythonFile(file2))

    def test_python_shebang(self):
        if False:
            for i in range(10):
                print('nop')
        file1 = os.path.join(self.test_tmpdir, 'testfile1')
        with open(file1, 'w') as f:
            f.write('#!/usr/bin/python\n')
        self.assertTrue(file_resources.IsPythonFile(file1))
        file2 = os.path.join(self.test_tmpdir, 'testfile2.run')
        with open(file2, 'w') as f:
            f.write('#! /bin/python2\n')
        self.assertTrue(file_resources.IsPythonFile(file1))

    def test_with_latin_encoding(self):
        if False:
            i = 10
            return i + 15
        file1 = os.path.join(self.test_tmpdir, 'testfile1')
        with codecs.open(file1, mode='w', encoding='latin-1') as f:
            f.write('#! /bin/python2\n')
        self.assertTrue(file_resources.IsPythonFile(file1))

    def test_with_invalid_encoding(self):
        if False:
            print('Hello World!')
        file1 = os.path.join(self.test_tmpdir, 'testfile1')
        with open(file1, 'w') as f:
            f.write('#! /bin/python2\n')
            f.write('# -*- coding: iso-3-14159 -*-\n')
        self.assertFalse(file_resources.IsPythonFile(file1))

class IsIgnoredTest(yapf_test_helper.YAPFTest):

    def test_root_path(self):
        if False:
            print('Hello World!')
        self.assertTrue(file_resources.IsIgnored('media', ['media']))
        self.assertFalse(file_resources.IsIgnored('media', ['media/*']))

    def test_sub_path(self):
        if False:
            return 10
        self.assertTrue(file_resources.IsIgnored('media/a', ['*/a']))
        self.assertTrue(file_resources.IsIgnored('media/b', ['media/*']))
        self.assertTrue(file_resources.IsIgnored('media/b/c', ['*/*/c']))

    def test_trailing_slash(self):
        if False:
            print('Hello World!')
        self.assertTrue(file_resources.IsIgnored('z', ['z']))
        self.assertTrue(file_resources.IsIgnored('z', ['z' + os.path.sep]))

class BufferedByteStream(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.stream = BytesIO()

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        return self.stream.getvalue().decode('utf-8')

    @property
    def buffer(self):
        if False:
            print('Hello World!')
        return self.stream

class WriteReformattedCodeTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.test_tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        shutil.rmtree(cls.test_tmpdir)

    def test_write_to_file(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'foobar\n'
        with utils.NamedTempFile(dirname=self.test_tmpdir) as (f, fname):
            file_resources.WriteReformattedCode(fname, s, in_place=True, encoding='utf-8')
            f.flush()
            with open(fname) as f2:
                self.assertEqual(f2.read(), s)

    def test_write_to_stdout(self):
        if False:
            return 10
        s = 'foobar'
        stream = BufferedByteStream()
        with utils.stdout_redirector(stream):
            file_resources.WriteReformattedCode(None, s, in_place=False, encoding='utf-8')
        self.assertEqual(stream.getvalue(), s)

    def test_write_encoded_to_stdout(self):
        if False:
            while True:
                i = 10
        s = '\ufeff# -*- coding: utf-8 -*-\nresult = "passed"\n'
        stream = BufferedByteStream()
        with utils.stdout_redirector(stream):
            file_resources.WriteReformattedCode(None, s, in_place=False, encoding='utf-8')
        self.assertEqual(stream.getvalue(), s)

class LineEndingTest(yapf_test_helper.YAPFTest):

    def test_line_ending_linefeed(self):
        if False:
            print('Hello World!')
        lines = ['spam\n', 'spam\n']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\n')

    def test_line_ending_carriage_return(self):
        if False:
            while True:
                i = 10
        lines = ['spam\r', 'spam\r']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\r')

    def test_line_ending_combo(self):
        if False:
            for i in range(10):
                print('nop')
        lines = ['spam\r\n', 'spam\r\n']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\r\n')

    def test_line_ending_weighted(self):
        if False:
            while True:
                i = 10
        lines = ['spam\n', 'spam\n', 'spam\r', 'spam\r\n']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\n')

    def test_line_ending_empty(self):
        if False:
            i = 10
            return i + 15
        lines = []
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\n')

    def test_line_ending_no_newline(self):
        if False:
            print('Hello World!')
        lines = ['spam']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\n')

    def test_line_ending_tie(self):
        if False:
            i = 10
            return i + 15
        lines = ['spam\n', 'spam\n', 'spam\r\n', 'spam\r\n']
        actual = file_resources.LineEnding(lines)
        self.assertEqual(actual, '\n')
if __name__ == '__main__':
    unittest.main()