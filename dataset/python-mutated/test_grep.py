from __future__ import absolute_import
import os
import re
import unicodedata as ud
from bzrlib import tests, osutils
from bzrlib._termcolor import color_string, FG
from bzrlib.tests.features import ColorFeature, UnicodeFilenameFeature

class GrepTestBase(tests.TestCaseWithTransport):
    """Base class for testing grep.

    Provides support methods for creating directory and file revisions.
    """
    _reflags = re.MULTILINE | re.DOTALL

    def _mk_file(self, path, line_prefix, total_lines, versioned):
        if False:
            print('Hello World!')
        text = ''
        for i in range(total_lines):
            text += line_prefix + str(i + 1) + '\n'
        open(path, 'w').write(text)
        if versioned:
            self.run_bzr(['add', path])
            self.run_bzr(['ci', '-m', '"' + path + '"'])

    def _update_file(self, path, text, checkin=True):
        if False:
            while True:
                i = 10
        "append text to file 'path' and check it in"
        open(path, 'a').write(text)
        if checkin:
            self.run_bzr(['ci', path, '-m', '"' + path + '"'])

    def _mk_unknown_file(self, path, line_prefix='line', total_lines=10):
        if False:
            for i in range(10):
                print('nop')
        self._mk_file(path, line_prefix, total_lines, versioned=False)

    def _mk_versioned_file(self, path, line_prefix='line', total_lines=10):
        if False:
            return 10
        self._mk_file(path, line_prefix, total_lines, versioned=True)

    def _mk_dir(self, path, versioned):
        if False:
            i = 10
            return i + 15
        os.mkdir(path)
        if versioned:
            self.run_bzr(['add', path])
            self.run_bzr(['ci', '-m', '"' + path + '"'])

    def _mk_unknown_dir(self, path):
        if False:
            while True:
                i = 10
        self._mk_dir(path, versioned=False)

    def _mk_versioned_dir(self, path):
        if False:
            print('Hello World!')
        self._mk_dir(path, versioned=True)

class TestGrep(GrepTestBase):
    """Core functional tests for grep."""

    def test_basic_unknown_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Search for pattern in specfic file.\n\n        If specified file is unknown, grep it anyway.'
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_unknown_file('file0.txt')
        (out, err) = self.run_bzr(['grep', 'line1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', 'line\\d+', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 10)
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', 'line1$'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)

    def test_ver_basic_file(self):
        if False:
            for i in range(10):
                print('nop')
        '(versioned) Search for pattern in specfic file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', '1', 'line1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~1:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '1', 'line[0-9]$', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~1:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 9)
        (out, err) = self.run_bzr(['grep', '-r', '1', 'line[0-9]', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~1:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 10)

    def test_wtree_basic_file(self):
        if False:
            i = 10
            return i + 15
        '(wtree) Search for pattern in specfic file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        self._update_file('file0.txt', 'ABC\n', checkin=False)
        (out, err) = self.run_bzr(['grep', 'ABC', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:ABC', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '[A-Z]{3}', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:ABC', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'ABC', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '[A-Z]{3}', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)

    def test_ver_basic_include(self):
        if False:
            print('Hello World!')
        '(versioned) Ensure that -I flag is respected.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.aa')
        self._mk_versioned_file('file0.bb')
        self._mk_versioned_file('file0.cc')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1$'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-I', '*.aa', '-I', '*.bb', 'line1'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-I', '*.aa', '-I', '*.bb', 'line1$'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_wtree_basic_include(self):
        if False:
            print('Hello World!')
        '(wtree) Ensure that --include flag is respected.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.aa')
        self._mk_versioned_file('file0.bb')
        self._mk_versioned_file('file0.cc')
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'line1'])
        self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'line1$'])
        self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_ver_basic_exclude(self):
        if False:
            print('Hello World!')
        '(versioned) Ensure that --exclude flag is respected.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.aa')
        self._mk_versioned_file('file0.bb')
        self._mk_versioned_file('file0.cc')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.cc', 'line1'])
        self.assertContainsRe(out, 'file0.aa~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.aa~.:line10', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line10', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.cc', 'line1$'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-X', '*.cc', 'line1'])
        self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)

    def test_wtree_basic_exclude(self):
        if False:
            while True:
                i = 10
        '(wtree) Ensure that --exclude flag is respected.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.aa')
        self._mk_versioned_file('file0.bb')
        self._mk_versioned_file('file0.cc')
        (out, err) = self.run_bzr(['grep', '--exclude', '*.cc', 'line1'])
        self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '--exclude', '*.cc', 'lin.1$'])
        self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_ver_multiple_files(self):
        if False:
            print('Hello World!')
        '(versioned) Search for pattern in multiple files.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_file('file2.txt', total_lines=2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line[1-2]$'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~.:line2', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:line2', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt~.:line2', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 6)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~.:line2', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:line2', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt~.:line2', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 6)

    def test_multiple_wtree_files(self):
        if False:
            print('Hello World!')
        '(wtree) Search for pattern in multiple files in working tree.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_file('file2.txt', total_lines=2)
        self._update_file('file0.txt', 'HELLO\n', checkin=False)
        self._update_file('file1.txt', 'HELLO\n', checkin=True)
        self._update_file('file2.txt', 'HELLO\n', checkin=False)
        (out, err) = self.run_bzr(['grep', 'HELLO', 'file0.txt', 'file1.txt', 'file2.txt'])
        self.assertContainsRe(out, 'file0.txt:HELLO', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt:HELLO', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt:HELLO', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', 'HELLO', '-r', 'last:1', 'file0.txt', 'file1.txt', 'file2.txt'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:HELLO', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file2.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', 'HE..O', 'file0.txt', 'file1.txt', 'file2.txt'])
        self.assertContainsRe(out, 'file0.txt:HELLO', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt:HELLO', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file2.txt:HELLO', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', 'HE..O', '-r', 'last:1', 'file0.txt', 'file1.txt', 'file2.txt'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~.:HELLO', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file2.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_ver_null_option(self):
        if False:
            print('Hello World!')
        '(versioned) --null option should use NUL instead of newline.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=3)
        nref = ud.normalize(u'NFC', u'file0.txt~1:line1\x00file0.txt~1:line2\x00file0.txt~1:line3\x00')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--null', 'line[1-3]'])
        nout = ud.normalize(u'NFC', out.decode('utf-8', 'ignore'))
        self.assertEqual(nout, nref)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-Z', 'line[1-3]'])
        nout = ud.normalize(u'NFC', out.decode('utf-8', 'ignore'))
        self.assertEqual(nout, nref)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--null', 'line'])
        nout = ud.normalize(u'NFC', out.decode('utf-8', 'ignore'))
        self.assertEqual(nout, nref)
        self.assertEqual(len(out.splitlines()), 1)

    def test_wtree_null_option(self):
        if False:
            return 10
        '(wtree) --null option should use NUL instead of newline.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=3)
        (out, err) = self.run_bzr(['grep', '--null', 'line[1-3]'])
        self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-Z', 'line[1-3]'])
        self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-Z', 'line'])
        self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
        self.assertEqual(len(out.splitlines()), 1)

    def test_versioned_file_in_dir_no_recursive(self):
        if False:
            for i in range(10):
                print('nop')
        '(versioned) Should not recurse with --no-recursive'
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('fileX.txt', line_prefix='lin')
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1$'])
        self.assertNotContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)

    def test_wtree_file_in_dir_no_recursive(self):
        if False:
            return 10
        '(wtree) Should not recurse with --no-recursive'
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('fileX.txt', line_prefix='lin')
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        (out, err) = self.run_bzr(['grep', '--no-recursive', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '--no-recursive', 'lin.1'])
        self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)

    def test_versioned_file_in_dir_recurse(self):
        if False:
            while True:
                i = 10
        '(versioned) Should recurse by default.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', '-1', '.i.e1'])
        self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '-1', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_wtree_file_in_dir_recurse(self):
        if False:
            print('Hello World!')
        '(wtree) Should recurse by default.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', 'lin.1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_versioned_file_within_dir(self):
        if False:
            for i in range(10):
                print('nop')
        '(versioned) Search for pattern while in nested dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line1'])
        self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '.i.e1'])
        self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_versioned_include_file_within_dir(self):
        if False:
            return 10
        '(versioned) Ensure --include is respected with file within dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_file('dir0/file1.aa')
        self._update_file('dir0/file1.aa', 'hello\n')
        self._update_file('dir0/file0.txt', 'hello\n')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', 'line1'])
        self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:2..last:1', '--include', '*.aa', 'line1'])
        self.assertContainsRe(out, '^file1.aa~4:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~4:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', 'lin.1'])
        self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:3..last:1', '--include', '*.aa', 'lin.1'])
        self.assertContainsRe(out, '^file1.aa~3:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~4:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~3:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~4:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 6)

    def test_versioned_exclude_file_within_dir(self):
        if False:
            for i in range(10):
                print('nop')
        '(versioned) Ensure --exclude is respected with file within dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_file('dir0/file1.aa')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.txt', 'line1'])
        self.assertContainsRe(out, '^file1.aa~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.txt', 'l[a-z]ne1'])
        self.assertContainsRe(out, '^file1.aa~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_wtree_file_within_dir(self):
        if False:
            while True:
                i = 10
        '(wtree) Search for pattern while in nested dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertContainsRe(out, '^file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', 'l[aeiou]ne1'])
        self.assertContainsRe(out, '^file0.txt:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_wtree_include_file_within_dir(self):
        if False:
            return 10
        '(wtree) Ensure --include is respected with file within dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_file('dir0/file1.aa')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', 'line1'])
        self.assertContainsRe(out, '^file1.aa:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', 'l[ixn]ne1'])
        self.assertContainsRe(out, '^file1.aa:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_wtree_exclude_file_within_dir(self):
        if False:
            print('Hello World!')
        '(wtree) Ensure --exclude is respected with file within dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_file('dir0/file1.aa')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '--exclude', '*.txt', 'li.e1'])
        self.assertContainsRe(out, '^file1.aa:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '--exclude', '*.txt', 'line1'])
        self.assertContainsRe(out, '^file1.aa:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.aa:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

    def test_versioned_include_from_outside_dir(self):
        if False:
            while True:
                i = 10
        '(versioned) Ensure --include is respected during recursive search.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.aa')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.bb')
        self._mk_versioned_dir('dir2')
        self._mk_versioned_file('dir2/file2.cc')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'l..e1'])
        self.assertContainsRe(out, '^dir0/file0.aa~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.aa~.:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.aa~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.aa~.:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)

    def test_wtree_include_from_outside_dir(self):
        if False:
            for i in range(10):
                print('nop')
        '(wtree) Ensure --include is respected during recursive search.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.aa')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.bb')
        self._mk_versioned_dir('dir2')
        self._mk_versioned_file('dir2/file2.cc')
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'l.n.1'])
        self.assertContainsRe(out, '^dir0/file0.aa:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.aa:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.aa:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.aa:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line10$', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)

    def test_versioned_exclude_from_outside_dir(self):
        if False:
            return 10
        '(versioned) Ensure --exclude is respected during recursive search.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.aa')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.bb')
        self._mk_versioned_dir('dir2')
        self._mk_versioned_file('dir2/file2.cc')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.cc', 'l..e1'])
        self.assertContainsRe(out, '^dir0/file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--exclude', '*.cc', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.aa~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb~.:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)

    def test_wtree_exclude_from_outside_dir(self):
        if False:
            return 10
        '(wtree) Ensure --exclude is respected during recursive search.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.aa')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.bb')
        self._mk_versioned_dir('dir2')
        self._mk_versioned_file('dir2/file2.cc')
        (out, err) = self.run_bzr(['grep', '--exclude', '*.cc', 'l[hijk]ne1'])
        self.assertContainsRe(out, '^dir0/file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--exclude', '*.cc', 'line1'])
        self.assertContainsRe(out, '^dir0/file0.aa:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.bb:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)

    def test_workingtree_files_from_outside_dir(self):
        if False:
            i = 10
            return i + 15
        '(wtree) Grep for pattern with dirs passed as argument.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.txt')
        (out, err) = self.run_bzr(['grep', 'l[aeiou]ne1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'line1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt:line1', flags=TestGrep._reflags)

    def test_versioned_files_from_outside_dir(self):
        if False:
            print('Hello World!')
        '(versioned) Grep for pattern with dirs passed as argument.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '.ine1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt~.:line1', flags=TestGrep._reflags)

    def test_wtree_files_from_outside_dir(self):
        if False:
            print('Hello World!')
        '(wtree) Grep for pattern with dirs passed as argument.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.txt')
        (out, err) = self.run_bzr(['grep', 'li.e1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'line1', 'dir0', 'dir1'])
        self.assertContainsRe(out, '^dir0/file0.txt:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir1/file1.txt:line1', flags=TestGrep._reflags)

    def test_versioned_files_from_outside_two_dirs(self):
        if False:
            while True:
                i = 10
        '(versioned) Grep for pattern with two levels of nested dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.txt')
        self._mk_versioned_dir('dir0/dir00')
        self._mk_versioned_file('dir0/dir00/file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'l.ne1', 'dir0/dir00'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'l.ne1'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line1', 'dir0/dir00'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line1'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt~.:line1', flags=TestGrep._reflags)

    def test_wtree_files_from_outside_two_dirs(self):
        if False:
            while True:
                i = 10
        '(wtree) Grep for pattern with two levels of nested dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._mk_versioned_dir('dir1')
        self._mk_versioned_file('dir1/file1.txt')
        self._mk_versioned_dir('dir0/dir00')
        self._mk_versioned_file('dir0/dir00/file0.txt')
        (out, err) = self.run_bzr(['grep', 'lin.1', 'dir0/dir00'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'li.e1'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'line1', 'dir0/dir00'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertContainsRe(out, '^dir0/dir00/file0.txt:line1', flags=TestGrep._reflags)

    def test_versioned_file_within_dir_two_levels(self):
        if False:
            while True:
                i = 10
        '(versioned) Search for pattern while in nested dir (two levels).\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_dir('dir0/dir1')
        self._mk_versioned_file('dir0/dir1/file0.txt')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '.ine1'])
        self.assertContainsRe(out, '^dir1/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'l.ne1'])
        self.assertContainsRe(out, '^dir0/dir1/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'lin.1'])
        self.assertContainsRe(out, '^dir1/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'line1'])
        self.assertContainsRe(out, '^dir0/dir1/file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)

    def test_wtree_file_within_dir_two_levels(self):
        if False:
            print('Hello World!')
        '(wtree) Search for pattern while in nested dir (two levels).\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_dir('dir0/dir1')
        self._mk_versioned_file('dir0/dir1/file0.txt')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', 'l[hij]ne1'])
        self.assertContainsRe(out, '^dir1/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--from-root', 'l.ne1'])
        self.assertContainsRe(out, '^dir0/dir1/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--no-recursive', 'lin.1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertContainsRe(out, '^dir1/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--from-root', 'line1'])
        self.assertContainsRe(out, '^dir0/dir1/file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--no-recursive', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)

    def test_versioned_ignore_case_no_match(self):
        if False:
            return 10
        '(versioned) Match fails without --ignore-case.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'LinE1', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'Li.E1', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)

    def test_wtree_ignore_case_no_match(self):
        if False:
            return 10
        '(wtree) Match fails without --ignore-case.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', 'LinE1', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '.inE1', 'file0.txt'])
        self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)

    def test_versioned_ignore_case_match(self):
        if False:
            print('Hello World!')
        '(versioned) Match fails without --ignore-case.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-i', 'Li.E1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-i', 'LinE1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--ignore-case', 'LinE1', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)

    def test_wtree_ignore_case_match(self):
        if False:
            i = 10
            return i + 15
        '(wtree) Match fails without --ignore-case.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '-i', 'LinE1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--ignore-case', 'LinE1', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--ignore-case', 'Li.E1', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt:line1', flags=TestGrep._reflags)

    def test_versioned_from_root_fail(self):
        if False:
            i = 10
            return i + 15
        '(versioned) Match should fail without --from-root.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        self._mk_versioned_dir('dir0')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'li.e1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)

    def test_wtree_from_root_fail(self):
        if False:
            while True:
                i = 10
        '(wtree) Match should fail without --from-root.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        self._mk_versioned_dir('dir0')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'li.e1'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)

    def test_versioned_from_root_pass(self):
        if False:
            print('Hello World!')
        '(versioned) Match pass with --from-root.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        self._mk_versioned_dir('dir0')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'l.ne1'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'line1'])
        self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)

    def test_wtree_from_root_pass(self):
        if False:
            for i in range(10):
                print('nop')
        '(wtree) Match pass with --from-root.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        self._mk_versioned_dir('dir0')
        os.chdir('dir0')
        (out, err) = self.run_bzr(['grep', '--from-root', 'lin.1'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--from-root', 'line1'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)

    def test_versioned_with_line_number(self):
        if False:
            i = 10
            return i + 15
        '(versioned) Search for pattern with --line-number.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--line-number', 'li.e3', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~.:3:line3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '--line-number', 'line3', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~.:3:line3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1', '-n', 'line1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt~.:1:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-n', 'line[0-9]', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)

    def test_wtree_with_line_number(self):
        if False:
            print('Hello World!')
        '(wtree) Search for pattern with --line-number.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt')
        (out, err) = self.run_bzr(['grep', '--line-number', 'line3', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-n', 'line1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:1:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-n', '[hjkl]ine1', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:1:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-n', 'line[0-9]', 'file0.txt'])
        self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)

    def test_revno_basic_history_grep_file(self):
        if False:
            print('Hello World!')
        'Search for pattern in specific revision number in a file.\n        '
        wd = 'foobar0'
        fname = 'file0.txt'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file(fname, total_lines=0)
        self._update_file(fname, text='v2 text\n')
        self._update_file(fname, text='v3 text\n')
        self._update_file(fname, text='v4 text\n')
        (out, err) = self.run_bzr(['grep', '-r', '2', 'v3', fname])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', 'v3', fname])
        self.assertContainsRe(out, 'file0.txt~3:v3.*', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '-n', 'v3', fname])
        self.assertContainsRe(out, 'file0.txt~3:2:v3.*', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '2', '[tuv]3', fname])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '[tuv]3', fname])
        self.assertContainsRe(out, 'file0.txt~3:v3.*', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '-n', '[tuv]3', fname])
        self.assertContainsRe(out, 'file0.txt~3:2:v3.*', flags=TestGrep._reflags)

    def test_revno_basic_history_grep_full(self):
        if False:
            return 10
        'Search for pattern in specific revision number in a file.\n        '
        wd = 'foobar0'
        fname = 'file0.txt'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file(fname, total_lines=0)
        self._mk_versioned_file('file1.txt')
        self._update_file(fname, text='v3 text\n')
        self._update_file(fname, text='v4 text\n')
        self._update_file(fname, text='v5 text\n')
        (out, err) = self.run_bzr(['grep', '-r', '2', 'v3'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '-n', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:1:v3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '2', '[tuv]3'])
        self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '[tuv]3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '3', '-n', '[tuv]3'])
        self.assertContainsRe(out, 'file0.txt~3:1:v3', flags=TestGrep._reflags)

    def test_revno_versioned_file_in_dir(self):
        if False:
            return 10
        'Grep specific version of file withing dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._update_file('dir0/file0.txt', 'v3 text\n')
        self._update_file('dir0/file0.txt', 'v4 text\n')
        self._update_file('dir0/file0.txt', 'v5 text\n')
        (out, err) = self.run_bzr(['grep', '-r', 'last:3', 'v4'])
        self.assertNotContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:2', 'v4'])
        self.assertContainsRe(out, '^dir0/file0.txt~4:v4', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:3', '[tuv]4'])
        self.assertNotContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', 'last:2', '[tuv]4'])
        self.assertContainsRe(out, '^dir0/file0.txt~4:v4', flags=TestGrep._reflags)

    def test_revno_range_basic_history_grep(self):
        if False:
            for i in range(10):
                print('nop')
        'Search for pattern in revision range for file.\n        '
        wd = 'foobar0'
        fname = 'file0.txt'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file(fname, total_lines=0)
        self._mk_versioned_file('file1.txt')
        self._update_file(fname, text='v3 text\n')
        self._update_file(fname, text='v4 text\n')
        self._update_file(fname, text='v5 text\n')
        self._update_file(fname, text='v6 text\n')
        (out, err) = self.run_bzr(['grep', '-r', '1..', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', '..1', 'v3'])
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '-r', '..6', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', '..', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', '1..5', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '5..1', 'v3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '1..', '[tuv]3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', '1..5', '[tuv]3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '5..1', '[tuv]3'])
        self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, 'file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)

    def test_revno_range_versioned_file_in_dir(self):
        if False:
            while True:
                i = 10
        'Grep rev-range for pattern for file withing a dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._update_file('dir0/file0.txt', 'v3 text\n')
        self._update_file('dir0/file0.txt', 'v4 text\n')
        self._update_file('dir0/file0.txt', 'v5 text\n')
        self._update_file('dir0/file0.txt', 'v6 text\n')
        (out, err) = self.run_bzr(['grep', '-r', '2..5', 'v3'])
        self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '2..5', '[tuv]3'])
        self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)

    def test_revno_range_versioned_file_from_outside_dir(self):
        if False:
            while True:
                i = 10
        'Grep rev-range for pattern from outside dir.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._update_file('dir0/file0.txt', 'v3 text\n')
        self._update_file('dir0/file0.txt', 'v4 text\n')
        self._update_file('dir0/file0.txt', 'v5 text\n')
        self._update_file('dir0/file0.txt', 'v6 text\n')
        (out, err) = self.run_bzr(['grep', '-r', '2..5', 'v3', 'dir0'])
        self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '-r', '2..5', '[tuv]3', 'dir0'])
        self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
        self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)

    def test_levels(self):
        if False:
            i = 10
            return i + 15
        '--levels=0 should show findings from merged revision.\n        '
        wd0 = 'foobar0'
        wd1 = 'foobar1'
        self.make_branch_and_tree(wd0)
        os.chdir(wd0)
        self._mk_versioned_file('file0.txt')
        os.chdir('..')
        (out, err) = self.run_bzr(['branch', wd0, wd1])
        os.chdir(wd1)
        self._mk_versioned_file('file1.txt')
        os.chdir(osutils.pathjoin('..', wd0))
        (out, err) = self.run_bzr(['merge', osutils.pathjoin('..', wd1)])
        (out, err) = self.run_bzr(['ci', '-m', 'merged'])
        (out, err) = self.run_bzr(['grep', 'line1'])
        self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt:line1', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', '--levels=0', 'line1'])
        self.assertContainsRe(out, '^file0.txt:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt:line10$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1..', '--levels=0', 'line1'])
        self.assertContainsRe(out, '^file0.txt~2:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~2:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:line10$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 8)
        (out, err) = self.run_bzr(['grep', '-r', '-1..', '-n', '--levels=0', 'line1'])
        self.assertContainsRe(out, '^file0.txt~2:1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~2:10:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:10:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:10:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:10:line10$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 8)
        (out, err) = self.run_bzr(['grep', '--levels=0', 'l.ne1'])
        self.assertContainsRe(out, '^file0.txt:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt:line10$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 4)
        (out, err) = self.run_bzr(['grep', '-r', 'last:1..', '--levels=0', 'lin.1'])
        self.assertContainsRe(out, '^file0.txt~2:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:line1$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~2:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~2:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~1.1.1:line10$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~1.1.1:line10$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 8)
        (out, err) = self.run_bzr(['grep', '-r', '-1..', '-n', '--levels=0', '.ine1'])
        self.assertContainsRe(out, 'file0.txt~2:1:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~2:1:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file0.txt~1.1.1:1:line1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.1:1:line1', flags=TestGrep._reflags)

    def test_dotted_rev_grep(self):
        if False:
            while True:
                i = 10
        'Grep in dotted revs\n        '
        wd0 = 'foobar0'
        wd1 = 'foobar1'
        self.make_branch_and_tree(wd0)
        os.chdir(wd0)
        self._mk_versioned_file('file0.txt')
        os.chdir('..')
        (out, err) = self.run_bzr(['branch', wd0, wd1])
        os.chdir(wd1)
        self._mk_versioned_file('file1.txt')
        self._update_file('file1.txt', 'text 0\n')
        self._update_file('file1.txt', 'text 1\n')
        self._update_file('file1.txt', 'text 2\n')
        os.chdir(osutils.pathjoin('..', wd0))
        (out, err) = self.run_bzr(['merge', osutils.pathjoin('..', wd1)])
        (out, err) = self.run_bzr(['ci', '-m', 'merged'])
        (out, err) = self.run_bzr(['grep', '-r', '1.1.1..1.1.4', 'text'])
        self.assertContainsRe(out, 'file1.txt~1.1.2:text 0', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.3:text 1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.3:text 1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.4:text 0', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.4:text 1', flags=TestGrep._reflags)
        self.assertContainsRe(out, 'file1.txt~1.1.4:text 2', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 6)

    def test_versioned_binary_file_grep(self):
        if False:
            return 10
        '(versioned) Grep for pattern in binary file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file.txt')
        self._mk_versioned_file('file0.bin')
        self._update_file('file0.bin', '\x00lineNN\x00\n')
        (out, err) = self.run_bzr(['grep', '-v', '-r', 'last:1', 'lineNN', 'file0.bin'])
        self.assertNotContainsRe(out, 'file0.bin', flags=TestGrep._reflags)
        self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        self.assertEqual(len(err.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-v', '-r', 'last:1', 'line.N', 'file0.bin'])
        self.assertNotContainsRe(out, 'file0.bin', flags=TestGrep._reflags)
        self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        self.assertEqual(len(err.splitlines()), 1)

    def test_wtree_binary_file_grep(self):
        if False:
            return 10
        '(wtree) Grep for pattern in binary file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.bin')
        self._update_file('file0.bin', '\x00lineNN\x00\n')
        (out, err) = self.run_bzr(['grep', '-v', 'lineNN', 'file0.bin'])
        self.assertNotContainsRe(out, 'file0.bin:line1', flags=TestGrep._reflags)
        self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
        (out, err) = self.run_bzr(['grep', 'lineNN', 'file0.bin'])
        self.assertNotContainsRe(out, 'file0.bin:line1', flags=TestGrep._reflags)
        self.assertNotContainsRe(err, 'Binary file', flags=TestGrep._reflags)

    def test_revspec(self):
        if False:
            return 10
        'Ensure various revspecs work\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file0.txt')
        self._update_file('dir0/file0.txt', 'v3 text\n')
        self._update_file('dir0/file0.txt', 'v4 text\n')
        self._update_file('dir0/file0.txt', 'v5 text\n')
        (out, err) = self.run_bzr(['grep', '-r', 'revno:1..2', 'v3'])
        self.assertNotContainsRe(out, 'file0', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '-r', 'revno:4..', 'v4'])
        self.assertContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '..revno:3', 'v4'])
        self.assertNotContainsRe(out, 'file0', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 0)
        (out, err) = self.run_bzr(['grep', '-r', '..revno:3', 'v3'])
        self.assertContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_wtree_files_with_matches(self):
        if False:
            print('Hello World!')
        '(wtree) Ensure --files-with-matches, -l works\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file00.txt', total_lines=2)
        self._mk_versioned_file('dir0/file01.txt', total_lines=2)
        self._update_file('file0.txt', 'HELLO\n', checkin=False)
        self._update_file('dir0/file00.txt', 'HELLO\n', checkin=False)
        (out, err) = self.run_bzr(['grep', '--files-with-matches', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '--files-with-matches', 'HE.LO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-l', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-l', 'HE.LO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-l', 'HELLO', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', '.ELLO', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', 'HELLO', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', '.ELLO', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-l', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-l', '.ELLO'])
        self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_ver_files_with_matches(self):
        if False:
            i = 10
            return i + 15
        '(ver) Ensure --files-with-matches, -l works\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file00.txt', total_lines=2)
        self._mk_versioned_file('dir0/file01.txt', total_lines=2)
        self._update_file('file0.txt', 'HELLO\n')
        self._update_file('dir0/file00.txt', 'HELLO\n')
        (out, err) = self.run_bzr(['grep', '-r', '-1', '--files-with-matches', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '--files-with-matches', 'H.LLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '6..7', '--files-with-matches', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '6..7', '--files-with-matches', 'H.LLO'])
        self.assertContainsRe(out, '^file0.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 3)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '-l', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '-l', 'H.LLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-l', 'HELLO', '-r', '-1', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', 'H.LLO', '-r', '-1', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^dir0/file00.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', 'HELLO', '-r', '-2', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt~6$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-l', 'HE.LO', '-r', '-2', 'file0.txt'])
        self.assertContainsRe(out, '^file0.txt~6$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-r', '-1', '-l', 'HELLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-r', '-1', '-l', '.ELLO'])
        self.assertContainsRe(out, '^file0.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_wtree_files_without_matches(self):
        if False:
            for i in range(10):
                print('nop')
        '(wtree) Ensure --files-without-match, -L works\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file00.txt', total_lines=2)
        self._mk_versioned_file('dir0/file01.txt', total_lines=2)
        self._update_file('file0.txt', 'HELLO\n', checkin=False)
        self._update_file('dir0/file00.txt', 'HELLO\n', checkin=False)
        (out, err) = self.run_bzr(['grep', '--files-without-match', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '--files-without-match', 'HE.LO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HE.LO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HELLO', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', '.ELLO', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HELLO', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-L', '.ELLO', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-L', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-L', '.ELLO'])
        self.assertContainsRe(out, '^file1.txt$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_ver_files_without_matches(self):
        if False:
            for i in range(10):
                print('nop')
        '(ver) Ensure --files-without-match, -L works\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        self._mk_versioned_file('file0.txt', total_lines=2)
        self._mk_versioned_file('file1.txt', total_lines=2)
        self._mk_versioned_dir('dir0')
        self._mk_versioned_file('dir0/file00.txt', total_lines=2)
        self._mk_versioned_file('dir0/file01.txt', total_lines=2)
        self._update_file('file0.txt', 'HELLO\n')
        self._update_file('dir0/file00.txt', 'HELLO\n')
        (out, err) = self.run_bzr(['grep', '-r', '-1', '--files-without-match', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '--files-without-match', 'H.LLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '6..7', '--files-without-match', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 5)
        (out, err) = self.run_bzr(['grep', '-r', '6..7', '--files-without-match', 'H.LLO'])
        self.assertContainsRe(out, '^file1.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file00.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~6$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 5)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '-L', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-r', '-1', '-L', 'H.LLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HELLO', '-r', '-1', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'H.LLO', '-r', '-1', 'dir0', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertContainsRe(out, '^dir0/file01.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)
        (out, err) = self.run_bzr(['grep', '-L', 'HELLO', '-r', '-2', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt~6$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-L', 'HE.LO', '-r', '-2', 'file1.txt'])
        self.assertContainsRe(out, '^file1.txt~6$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-r', '-1', '-L', 'HELLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--no-recursive', '-r', '-1', '-L', '.ELLO'])
        self.assertContainsRe(out, '^file1.txt~7$', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 1)

    def test_no_tree(self):
        if False:
            print('Hello World!')
        'Ensure grep works without working tree.\n        '
        wd0 = 'foobar0'
        wd1 = 'foobar1'
        self.make_branch_and_tree(wd0)
        os.chdir(wd0)
        self._mk_versioned_file('file0.txt')
        os.chdir('..')
        (out, err) = self.run_bzr(['branch', '--no-tree', wd0, wd1])
        os.chdir(wd1)
        (out, err) = self.run_bzr(['grep', 'line1'], 3)
        self.assertContainsRe(err, 'Cannot search working tree', flags=TestGrep._reflags)
        self.assertEqual(out, '')
        (out, err) = self.run_bzr(['grep', '-r', '1', 'line1'])
        self.assertContainsRe(out, 'file0.txt~1:line1', flags=TestGrep._reflags)
        self.assertEqual(len(out.splitlines()), 2)

class TestNonAscii(GrepTestBase):
    """Tests for non-ascii filenames and file contents"""
    _test_needs_features = [UnicodeFilenameFeature]

    def test_unicode_only_file(self):
        if False:
            i = 10
            return i + 15
        'Test filename and contents that requires a unicode encoding'
        tree = self.make_branch_and_tree('.')
        contents = [u'ሴ']
        self.build_tree(contents)
        tree.add(contents)
        tree.commit('Initial commit')
        as_utf8 = u'ሴ'.encode('UTF-8')
        streams = self.run_bzr(['grep', '--files-with-matches', u'contents'], encoding='UTF-8')
        self.assertEqual(streams, (as_utf8 + '\n', ''))
        streams = self.run_bzr(['grep', '-r', '1', '--files-with-matches', u'contents'], encoding='UTF-8')
        self.assertEqual(streams, (as_utf8 + '~1\n', ''))
        fileencoding = osutils.get_user_encoding()
        as_mangled = as_utf8.decode(fileencoding, 'replace').encode('UTF-8')
        streams = self.run_bzr(['grep', '-n', u'contents'], encoding='UTF-8')
        self.assertEqual(streams, ('%s:1:contents of %s\n' % (as_utf8, as_mangled), ''))
        streams = self.run_bzr(['grep', '-n', '-r', '1', u'contents'], encoding='UTF-8')
        self.assertEqual(streams, ('%s~1:1:contents of %s\n' % (as_utf8, as_mangled), ''))

class TestColorGrep(GrepTestBase):
    """Tests for the --color option."""
    _test_needs_features = [ColorFeature]
    _rev_sep = color_string('~', fg=FG.BOLD_YELLOW)
    _sep = color_string(':', fg=FG.BOLD_CYAN)

    def test_color_option(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure options for color are valid.\n        '
        (out, err) = self.run_bzr(['grep', '--color', 'foo', 'bar'], 3)
        self.assertEqual(out, '')
        self.assertContainsRe(err, 'Valid values for --color are', flags=TestGrep._reflags)

    def test_ver_matching_files(self):
        if False:
            print('Hello World!')
        '(versioned) Search for matches or no matches only'
        tree = self.make_branch_and_tree('.')
        contents = ['d/', 'd/aaa', 'bbb']
        self.build_tree(contents)
        tree.add(contents)
        tree.commit('Initial commit')
        streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-with-matches', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', self._rev_sep, '1', '\n']), ''))
        streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-without-match', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', self._rev_sep, '1', '\n']), ''))

    def test_wtree_matching_files(self):
        if False:
            print('Hello World!')
        '(wtree) Search for matches or no matches only'
        tree = self.make_branch_and_tree('.')
        contents = ['d/', 'd/aaa', 'bbb']
        self.build_tree(contents)
        tree.add(contents)
        tree.commit('Initial commit')
        streams = self.run_bzr(['grep', '--color', 'always', '--files-with-matches', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', FG.NONE, '\n']), ''))
        streams = self.run_bzr(['grep', '--color', 'always', '--files-without-match', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', FG.NONE, '\n']), ''))

    def test_ver_basic_file(self):
        if False:
            i = 10
            return i + 15
        '(versioned) Search for pattern in specfic file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        lp = 'foo is foobar'
        self._mk_versioned_file('file0.txt', line_prefix=lp, total_lines=1)
        foo = color_string('foo', fg=FG.BOLD_RED)
        res = FG.MAGENTA + 'file0.txt' + self._rev_sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        txt_res = 'file0.txt~1:foo is foobar1\n'
        nres = FG.MAGENTA + 'file0.txt' + self._rev_sep + '1' + self._sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        (out, err) = self.run_bzr(['grep', '--color', 'always', '-r', '1', 'foo'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--color', 'auto', '-r', '1', 'foo'])
        self.assertEqual(out, txt_res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-i', '--color', 'always', '-r', '1', 'FOO'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--color', 'always', '-r', '1', 'f.o'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-i', '--color', 'always', '-r', '1', 'F.O'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '--color', 'always', '-r', '1', 'foo'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '-i', '--color', 'always', '-r', '1', 'FOO'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '--color', 'always', '-r', '1', 'f.o'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '-i', '--color', 'always', '-r', '1', 'F.O'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)

    def test_wtree_basic_file(self):
        if False:
            while True:
                i = 10
        '(wtree) Search for pattern in specfic file.\n        '
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        lp = 'foo is foobar'
        self._mk_versioned_file('file0.txt', line_prefix=lp, total_lines=1)
        foo = color_string('foo', fg=FG.BOLD_RED)
        res = FG.MAGENTA + 'file0.txt' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        nres = FG.MAGENTA + 'file0.txt' + self._sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        (out, err) = self.run_bzr(['grep', '--color', 'always', 'foo'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-i', '--color', 'always', 'FOO'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '--color', 'always', 'f.o'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-i', '--color', 'always', 'F.O'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '--color', 'always', 'foo'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '-i', '--color', 'always', 'FOO'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '--color', 'always', 'f.o'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        (out, err) = self.run_bzr(['grep', '-n', '-i', '--color', 'always', 'F.O'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)

def subst_dates(string):
    if False:
        while True:
            i = 10
    'Replace date strings with constant values.'
    return re.sub('\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} [-\\+]\\d{4}', 'YYYY-MM-DD HH:MM:SS +ZZZZ', string)

class TestGrepDiff(tests.TestCaseWithTransport):

    def make_example_branch(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('hello', 'foo\n'), ('goodbye', 'baz\n')])
        tree.add(['hello'])
        tree.commit('setup')
        tree.add(['goodbye'])
        tree.commit('setup')
        return tree

    def test_grep_diff_basic(self):
        if False:
            while True:
                i = 10
        'grep -p basic test.'
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', 'hello world!\n')])
        tree.commit('updated hello')
        (out, err) = self.run_bzr(['grep', '-p', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n=== revno:1 ===\n  === added file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n")

    def test_grep_diff_revision(self):
        if False:
            while True:
                i = 10
        'grep -p specific revision.'
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', 'hello world!\n')])
        tree.commit('updated hello')
        (out, err) = self.run_bzr(['grep', '-p', '-r', '3', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n")

    def test_grep_diff_revision_range(self):
        if False:
            i = 10
            return i + 15
        'grep -p revision range.'
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', 'hello world!1\n')])
        tree.commit('rev3')
        self.build_tree_contents([('blah', 'hello world!2\n')])
        tree.add('blah')
        tree.commit('rev4')
        open('hello', 'a').write('hello world!3\n')
        tree.commit('rev5')
        (out, err) = self.run_bzr(['grep', '-p', '-r', '2..5', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:5 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!3\n=== revno:4 ===\n  === added file 'blah'\n    +hello world!2\n=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!1\n")

    def test_grep_diff_color(self):
        if False:
            i = 10
            return i + 15
        'grep -p color test.'
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', 'hello world!\n')])
        tree.commit('updated hello')
        (out, err) = self.run_bzr(['grep', '--diff', '-r', '3', '--color', 'always', 'hello'])
        self.assertEqual(err, '')
        revno = color_string('=== revno:3 ===', fg=FG.BOLD_BLUE) + '\n'
        filename = color_string("  === modified file 'hello'", fg=FG.BOLD_MAGENTA) + '\n'
        redhello = color_string('hello', fg=FG.BOLD_RED)
        diffstr = '    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n'
        diffstr = diffstr.replace('hello', redhello)
        self.assertEqualDiff(subst_dates(out), revno + filename + diffstr)

    def test_grep_norevs(self):
        if False:
            i = 10
            return i + 15
        'grep -p with zero revisions.'
        (out, err) = self.run_bzr(['init'])
        (out, err) = self.run_bzr(['grep', '--diff', 'foo'], 3)
        self.assertEqual(out, '')
        self.assertContainsRe(err, 'ERROR:.*revision.* does not exist in branch')