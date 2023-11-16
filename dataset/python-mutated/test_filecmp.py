import filecmp
import os
import shutil
import tempfile
import unittest
from test import support
from test.support import os_helper

class FileCompareTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.name = os_helper.TESTFN
        self.name_same = os_helper.TESTFN + '-same'
        self.name_diff = os_helper.TESTFN + '-diff'
        data = 'Contents of file go here.\n'
        for name in [self.name, self.name_same, self.name_diff]:
            with open(name, 'w', encoding='utf-8') as output:
                output.write(data)
        with open(self.name_diff, 'a+', encoding='utf-8') as output:
            output.write('An extra line.\n')
        self.dir = tempfile.gettempdir()

    def tearDown(self):
        if False:
            while True:
                i = 10
        os.unlink(self.name)
        os.unlink(self.name_same)
        os.unlink(self.name_diff)

    def test_matching(self):
        if False:
            while True:
                i = 10
        self.assertTrue(filecmp.cmp(self.name, self.name), 'Comparing file to itself fails')
        self.assertTrue(filecmp.cmp(self.name, self.name, shallow=False), 'Comparing file to itself fails')
        self.assertTrue(filecmp.cmp(self.name, self.name_same), 'Comparing file to identical file fails')
        self.assertTrue(filecmp.cmp(self.name, self.name_same, shallow=False), 'Comparing file to identical file fails')

    def test_different(self):
        if False:
            return 10
        self.assertFalse(filecmp.cmp(self.name, self.name_diff), 'Mismatched files compare as equal')
        self.assertFalse(filecmp.cmp(self.name, self.dir), 'File and directory compare as equal')

    def test_cache_clear(self):
        if False:
            for i in range(10):
                print('nop')
        first_compare = filecmp.cmp(self.name, self.name_same, shallow=False)
        second_compare = filecmp.cmp(self.name, self.name_diff, shallow=False)
        filecmp.clear_cache()
        self.assertTrue(len(filecmp._cache) == 0, 'Cache not cleared after calling clear_cache')

class DirCompareTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        tmpdir = tempfile.gettempdir()
        self.dir = os.path.join(tmpdir, 'dir')
        self.dir_same = os.path.join(tmpdir, 'dir-same')
        self.dir_diff = os.path.join(tmpdir, 'dir-diff')
        self.dir_ignored = os.path.join(self.dir_same, '.hg')
        self.caseinsensitive = os.path.normcase('A') == os.path.normcase('a')
        data = 'Contents of file go here.\n'
        for dir in (self.dir, self.dir_same, self.dir_diff, self.dir_ignored):
            shutil.rmtree(dir, True)
            os.mkdir(dir)
            subdir_path = os.path.join(dir, 'subdir')
            os.mkdir(subdir_path)
            if self.caseinsensitive and dir is self.dir_same:
                fn = 'FiLe'
            else:
                fn = 'file'
            with open(os.path.join(dir, fn), 'w', encoding='utf-8') as output:
                output.write(data)
        with open(os.path.join(self.dir_diff, 'file2'), 'w', encoding='utf-8') as output:
            output.write('An extra file.\n')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        for dir in (self.dir, self.dir_same, self.dir_diff):
            shutil.rmtree(dir)

    def test_default_ignores(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIn('.hg', filecmp.DEFAULT_IGNORES)

    def test_cmpfiles(self):
        if False:
            print('Hello World!')
        self.assertTrue(filecmp.cmpfiles(self.dir, self.dir, ['file']) == (['file'], [], []), 'Comparing directory to itself fails')
        self.assertTrue(filecmp.cmpfiles(self.dir, self.dir_same, ['file']) == (['file'], [], []), 'Comparing directory to same fails')
        self.assertTrue(filecmp.cmpfiles(self.dir, self.dir, ['file'], shallow=False) == (['file'], [], []), 'Comparing directory to itself fails')
        self.assertTrue(filecmp.cmpfiles(self.dir, self.dir_same, ['file'], shallow=False), 'Comparing directory to same fails')
        with open(os.path.join(self.dir, 'file2'), 'w', encoding='utf-8') as output:
            output.write('Different contents.\n')
        self.assertFalse(filecmp.cmpfiles(self.dir, self.dir_same, ['file', 'file2']) == (['file'], ['file2'], []), 'Comparing mismatched directories fails')

    def _assert_lists(self, actual, expected):
        if False:
            print('Hello World!')
        'Assert that two lists are equal, up to ordering.'
        self.assertEqual(sorted(actual), sorted(expected))

    def test_dircmp(self):
        if False:
            i = 10
            return i + 15
        (left_dir, right_dir) = (self.dir, self.dir_same)
        d = filecmp.dircmp(left_dir, right_dir)
        self.assertEqual(d.left, left_dir)
        self.assertEqual(d.right, right_dir)
        if self.caseinsensitive:
            self._assert_lists(d.left_list, ['file', 'subdir'])
            self._assert_lists(d.right_list, ['FiLe', 'subdir'])
        else:
            self._assert_lists(d.left_list, ['file', 'subdir'])
            self._assert_lists(d.right_list, ['file', 'subdir'])
        self._assert_lists(d.common, ['file', 'subdir'])
        self._assert_lists(d.common_dirs, ['subdir'])
        self.assertEqual(d.left_only, [])
        self.assertEqual(d.right_only, [])
        self.assertEqual(d.same_files, ['file'])
        self.assertEqual(d.diff_files, [])
        expected_report = ['diff {} {}'.format(self.dir, self.dir_same), "Identical files : ['file']", "Common subdirectories : ['subdir']"]
        self._assert_report(d.report, expected_report)
        (left_dir, right_dir) = (self.dir, self.dir_diff)
        d = filecmp.dircmp(left_dir, right_dir)
        self.assertEqual(d.left, left_dir)
        self.assertEqual(d.right, right_dir)
        self._assert_lists(d.left_list, ['file', 'subdir'])
        self._assert_lists(d.right_list, ['file', 'file2', 'subdir'])
        self._assert_lists(d.common, ['file', 'subdir'])
        self._assert_lists(d.common_dirs, ['subdir'])
        self.assertEqual(d.left_only, [])
        self.assertEqual(d.right_only, ['file2'])
        self.assertEqual(d.same_files, ['file'])
        self.assertEqual(d.diff_files, [])
        expected_report = ['diff {} {}'.format(self.dir, self.dir_diff), "Only in {} : ['file2']".format(self.dir_diff), "Identical files : ['file']", "Common subdirectories : ['subdir']"]
        self._assert_report(d.report, expected_report)
        (left_dir, right_dir) = (self.dir, self.dir_diff)
        shutil.move(os.path.join(self.dir_diff, 'file2'), os.path.join(self.dir, 'file2'))
        d = filecmp.dircmp(left_dir, right_dir)
        self.assertEqual(d.left, left_dir)
        self.assertEqual(d.right, right_dir)
        self._assert_lists(d.left_list, ['file', 'file2', 'subdir'])
        self._assert_lists(d.right_list, ['file', 'subdir'])
        self._assert_lists(d.common, ['file', 'subdir'])
        self.assertEqual(d.left_only, ['file2'])
        self.assertEqual(d.right_only, [])
        self.assertEqual(d.same_files, ['file'])
        self.assertEqual(d.diff_files, [])
        expected_report = ['diff {} {}'.format(self.dir, self.dir_diff), "Only in {} : ['file2']".format(self.dir), "Identical files : ['file']", "Common subdirectories : ['subdir']"]
        self._assert_report(d.report, expected_report)
        with open(os.path.join(self.dir_diff, 'file2'), 'w', encoding='utf-8') as output:
            output.write('Different contents.\n')
        d = filecmp.dircmp(self.dir, self.dir_diff)
        self.assertEqual(d.same_files, ['file'])
        self.assertEqual(d.diff_files, ['file2'])
        expected_report = ['diff {} {}'.format(self.dir, self.dir_diff), "Identical files : ['file']", "Differing files : ['file2']", "Common subdirectories : ['subdir']"]
        self._assert_report(d.report, expected_report)

    def test_dircmp_subdirs_type(self):
        if False:
            while True:
                i = 10
        'Check that dircmp.subdirs respects subclassing.'

        class MyDirCmp(filecmp.dircmp):
            pass
        d = MyDirCmp(self.dir, self.dir_diff)
        sub_dirs = d.subdirs
        self.assertEqual(list(sub_dirs.keys()), ['subdir'])
        sub_dcmp = sub_dirs['subdir']
        self.assertEqual(type(sub_dcmp), MyDirCmp)

    def test_report_partial_closure(self):
        if False:
            for i in range(10):
                print('nop')
        (left_dir, right_dir) = (self.dir, self.dir_same)
        d = filecmp.dircmp(left_dir, right_dir)
        left_subdir = os.path.join(left_dir, 'subdir')
        right_subdir = os.path.join(right_dir, 'subdir')
        expected_report = ['diff {} {}'.format(self.dir, self.dir_same), "Identical files : ['file']", "Common subdirectories : ['subdir']", '', 'diff {} {}'.format(left_subdir, right_subdir)]
        self._assert_report(d.report_partial_closure, expected_report)

    def test_report_full_closure(self):
        if False:
            while True:
                i = 10
        (left_dir, right_dir) = (self.dir, self.dir_same)
        d = filecmp.dircmp(left_dir, right_dir)
        left_subdir = os.path.join(left_dir, 'subdir')
        right_subdir = os.path.join(right_dir, 'subdir')
        expected_report = ['diff {} {}'.format(self.dir, self.dir_same), "Identical files : ['file']", "Common subdirectories : ['subdir']", '', 'diff {} {}'.format(left_subdir, right_subdir)]
        self._assert_report(d.report_full_closure, expected_report)

    def _assert_report(self, dircmp_report, expected_report_lines):
        if False:
            while True:
                i = 10
        with support.captured_stdout() as stdout:
            dircmp_report()
            report_lines = stdout.getvalue().strip().split('\n')
            self.assertEqual(report_lines, expected_report_lines)
if __name__ == '__main__':
    unittest.main()