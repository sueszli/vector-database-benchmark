"""Tests for the win32 walkdir extension."""
import errno
from bzrlib import osutils, tests
from bzrlib.tests import features
win32_readdir_feature = features.ModuleAvailableFeature('bzrlib._walkdirs_win32')

class TestWin32Finder(tests.TestCaseInTempDir):
    _test_needs_features = [win32_readdir_feature]

    def setUp(self):
        if False:
            return 10
        super(TestWin32Finder, self).setUp()
        from bzrlib._walkdirs_win32 import Win32ReadDir
        self.reader = Win32ReadDir()

    def _remove_stat_from_dirblock(self, dirblock):
        if False:
            i = 10
            return i + 15
        return [info[:3] + info[4:] for info in dirblock]

    def assertWalkdirs(self, expected, top, prefix=''):
        if False:
            for i in range(10):
                print('nop')
        old_selected_dir_reader = osutils._selected_dir_reader
        try:
            osutils._selected_dir_reader = self.reader
            finder = osutils._walkdirs_utf8(top, prefix=prefix)
            result = []
            for (dirname, dirblock) in finder:
                dirblock = self._remove_stat_from_dirblock(dirblock)
                result.append((dirname, dirblock))
            self.assertEqual(expected, result)
        finally:
            osutils._selected_dir_reader = old_selected_dir_reader

    def assertReadDir(self, expected, prefix, top_unicode):
        if False:
            print('Hello World!')
        result = self._remove_stat_from_dirblock(self.reader.read_dir(prefix, top_unicode))
        self.assertEqual(expected, result)

    def test_top_prefix_to_starting_dir(self):
        if False:
            print('Hello World!')
        self.assertEqual(('prefix', None, None, None, u'\x12'), self.reader.top_prefix_to_starting_dir(u'\x12'.encode('utf8'), 'prefix'))

    def test_empty_directory(self):
        if False:
            return 10
        self.assertReadDir([], 'prefix', u'.')
        self.assertWalkdirs([(('', u'.'), [])], u'.')

    def test_file(self):
        if False:
            print('Hello World!')
        self.build_tree(['foo'])
        self.assertReadDir([('foo', 'foo', 'file', u'./foo')], '', u'.')

    def test_directory(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(['bar/'])
        self.assertReadDir([('bar', 'bar', 'directory', u'./bar')], '', u'.')

    def test_prefix(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['bar/', 'baf'])
        self.assertReadDir([('xxx/baf', 'baf', 'file', u'./baf'), ('xxx/bar', 'bar', 'directory', u'./bar')], 'xxx', u'.')

    def test_missing_dir(self):
        if False:
            return 10
        e = self.assertRaises(WindowsError, self.reader.read_dir, 'prefix', u'no_such_dir')
        self.assertEqual(errno.ENOENT, e.errno)
        self.assertEqual(3, e.winerror)
        self.assertEqual((3, u'no_such_dir/*'), e.args)

class Test_Win32Stat(tests.TestCaseInTempDir):
    _test_needs_features = [win32_readdir_feature]

    def setUp(self):
        if False:
            return 10
        super(Test_Win32Stat, self).setUp()
        from bzrlib._walkdirs_win32 import lstat
        self.win32_lstat = lstat

    def test_zero_members_present(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['foo'])
        st = self.win32_lstat('foo')
        self.assertEqual(0, st.st_dev)
        self.assertEqual(0, st.st_ino)
        self.assertEqual(0, st.st_uid)
        self.assertEqual(0, st.st_gid)