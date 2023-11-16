"""Tests for win32utils."""
import os
from bzrlib import osutils, symbol_versioning, tests, win32utils
from bzrlib.tests import TestCase, TestCaseInTempDir, TestSkipped
from bzrlib.tests.features import backslashdir_feature
from bzrlib.win32utils import glob_expand, get_app_path
from bzrlib.tests import features
Win32RegistryFeature = features.ModuleAvailableFeature('_winreg')
CtypesFeature = features.ModuleAvailableFeature('ctypes')
Win32comShellFeature = features.ModuleAvailableFeature('win32com.shell')
Win32ApiFeature = features.ModuleAvailableFeature('win32api')

class TestWin32UtilsGlobExpand(TestCaseInTempDir):
    _test_needs_features = []

    def test_empty_tree(self):
        if False:
            print('Hello World!')
        self.build_tree([])
        self._run_testset([[['a'], ['a']], [['?'], ['?']], [['*'], ['*']], [['a', 'a'], ['a', 'a']]])

    def build_ascii_tree(self):
        if False:
            return 10
        self.build_tree(['a', 'a1', 'a2', 'a11', 'a.1', 'b', 'b1', 'b2', 'b3', 'c/', 'c/c1', 'c/c2', 'd/', 'd/d1', 'd/d2', 'd/e/', 'd/e/e1'])

    def build_unicode_tree(self):
        if False:
            print('Hello World!')
        self.requireFeature(features.UnicodeFilenameFeature)
        self.build_tree([u'ሴ', u'ሴሴ', u'ስ/', u'ስ/ስ'])

    def test_tree_ascii(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks the glob expansion and path separation char\n        normalization'
        self.build_ascii_tree()
        self._run_testset([[[u'a'], [u'a']], [[u'a', u'a'], [u'a', u'a']], [[u'd'], [u'd']], [[u'd/'], [u'd/']], [[u'a*'], [u'a', u'a1', u'a2', u'a11', u'a.1']], [[u'?'], [u'a', u'b', u'c', u'd']], [[u'a?'], [u'a1', u'a2']], [[u'a??'], [u'a11', u'a.1']], [[u'b[1-2]'], [u'b1', u'b2']], [[u'd/*'], [u'd/d1', u'd/d2', u'd/e']], [[u'?/*'], [u'c/c1', u'c/c2', u'd/d1', u'd/d2', u'd/e']], [[u'*/*'], [u'c/c1', u'c/c2', u'd/d1', u'd/d2', u'd/e']], [[u'*/'], [u'c/', u'd/']]])

    def test_backslash_globbing(self):
        if False:
            print('Hello World!')
        self.requireFeature(backslashdir_feature)
        self.build_ascii_tree()
        self._run_testset([[[u'd\\'], [u'd/']], [[u'd\\*'], [u'd/d1', u'd/d2', u'd/e']], [[u'?\\*'], [u'c/c1', u'c/c2', u'd/d1', u'd/d2', u'd/e']], [[u'*\\*'], [u'c/c1', u'c/c2', u'd/d1', u'd/d2', u'd/e']], [[u'*\\'], [u'c/', u'd/']]])

    def test_case_insensitive_globbing(self):
        if False:
            while True:
                i = 10
        if os.path.normcase('AbC') == 'AbC':
            self.skip('Test requires case insensitive globbing function')
        self.build_ascii_tree()
        self._run_testset([[[u'A'], [u'A']], [[u'A?'], [u'a1', u'a2']]])

    def test_tree_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks behaviour with non-ascii filenames'
        self.build_unicode_tree()
        self._run_testset([[[u'ሴ'], [u'ሴ']], [[u'ስ'], [u'ስ']], [[u'ስ/'], [u'ስ/']], [[u'ስ/ስ'], [u'ስ/ስ']], [[u'?'], [u'ሴ', u'ስ']], [[u'*'], [u'ሴ', u'ሴሴ', u'ስ']], [[u'ሴ*'], [u'ሴ', u'ሴሴ']], [[u'ስ/?'], [u'ስ/ስ']], [[u'ስ/*'], [u'ስ/ስ']], [[u'?/'], [u'ስ/']], [[u'*/'], [u'ስ/']], [[u'?/?'], [u'ስ/ስ']], [[u'*/*'], [u'ስ/ስ']]])

    def test_unicode_backslashes(self):
        if False:
            return 10
        self.requireFeature(backslashdir_feature)
        self.build_unicode_tree()
        self._run_testset([[[u'ስ\\'], [u'ስ/']], [[u'ስ\\ስ'], [u'ስ/ስ']], [[u'ስ\\?'], [u'ስ/ስ']], [[u'ስ\\*'], [u'ስ/ስ']], [[u'?\\'], [u'ስ/']], [[u'*\\'], [u'ስ/']], [[u'?\\?'], [u'ስ/ስ']], [[u'*\\*'], [u'ስ/ስ']]])

    def _run_testset(self, testset):
        if False:
            print('Hello World!')
        for (pattern, expected) in testset:
            result = glob_expand(pattern)
            expected.sort()
            result.sort()
            self.assertEqual(expected, result, 'pattern %s' % pattern)

class TestAppPaths(TestCase):
    _test_needs_features = [Win32RegistryFeature]

    def test_iexplore(self):
        if False:
            return 10
        for a in ('iexplore', 'iexplore.exe'):
            p = get_app_path(a)
            (d, b) = os.path.split(p)
            self.assertEqual('iexplore.exe', b.lower())
            self.assertNotEqual('', d)

    def test_wordpad(self):
        if False:
            while True:
                i = 10
        self.requireFeature(Win32ApiFeature)
        for a in ('wordpad', 'wordpad.exe'):
            p = get_app_path(a)
            (d, b) = os.path.split(p)
            self.assertEqual('wordpad.exe', b.lower())
            self.assertNotEqual('', d)

    def test_not_existing(self):
        if False:
            for i in range(10):
                print('nop')
        p = get_app_path('not-existing')
        self.assertEqual('not-existing', p)

class TestLocations(TestCase):
    """Tests for windows specific path and name retrieving functions"""

    def test__ensure_unicode_deprecated(self):
        if False:
            while True:
                i = 10
        s = 'text'
        u1 = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils._ensure_unicode, s)
        self.assertEqual(s, u1)
        self.assertIsInstance(u1, unicode)
        u2 = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils._ensure_unicode, u1)
        self.assertIs(u1, u2)

    def test_appdata_unicode_deprecated(self):
        if False:
            return 10
        self.overrideEnv('APPDATA', 'fakepath')
        s = win32utils.get_appdata_location()
        u = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils.get_appdata_location_unicode)
        self.assertEqual(s, u)
        self.assertIsInstance(s, unicode)

    def test_home_unicode_deprecated(self):
        if False:
            i = 10
            return i + 15
        s = win32utils.get_home_location()
        u = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils.get_home_location_unicode)
        self.assertEqual(s, u)
        self.assertIsInstance(s, unicode)

    def test_user_unicode_deprecated(self):
        if False:
            while True:
                i = 10
        self.overrideEnv('USERNAME', 'alien')
        s = win32utils.get_user_name()
        u = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils.get_user_name_unicode)
        self.assertEqual(s, u)
        self.assertIsInstance(s, unicode)

    def test_host_unicode_deprecated(self):
        if False:
            i = 10
            return i + 15
        self.overrideEnv('COMPUTERNAME', 'alienbox')
        s = win32utils.get_host_name()
        u = self.applyDeprecated(symbol_versioning.deprecated_in((2, 5, 0)), win32utils.get_host_name_unicode)
        self.assertEqual(s, u)
        self.assertIsInstance(s, unicode)

class TestLocationsCtypes(TestCase):
    _test_needs_features = [CtypesFeature]

    def assertPathsEqual(self, p1, p2):
        if False:
            return 10
        self.assertEqual(p1, p2)

    def test_appdata_not_using_environment(self):
        if False:
            i = 10
            return i + 15
        first = win32utils.get_appdata_location()
        self.overrideEnv('APPDATA', None)
        self.assertPathsEqual(first, win32utils.get_appdata_location())

    def test_appdata_matches_environment(self):
        if False:
            return 10
        encoding = osutils.get_user_encoding()
        env_val = os.environ.get('APPDATA', None)
        if not env_val:
            raise TestSkipped('No APPDATA environment variable exists')
        self.assertPathsEqual(win32utils.get_appdata_location(), env_val.decode(encoding))

    def test_local_appdata_not_using_environment(self):
        if False:
            while True:
                i = 10
        first = win32utils.get_local_appdata_location()
        self.overrideEnv('LOCALAPPDATA', None)
        self.assertPathsEqual(first, win32utils.get_local_appdata_location())

    def test_local_appdata_matches_environment(self):
        if False:
            return 10
        lad = win32utils.get_local_appdata_location()
        env = os.environ.get('LOCALAPPDATA')
        if env:
            encoding = osutils.get_user_encoding()
            self.assertPathsEqual(lad, env.decode(encoding))

class TestLocationsPywin32(TestLocationsCtypes):
    _test_needs_features = [Win32comShellFeature]

    def setUp(self):
        if False:
            return 10
        super(TestLocationsPywin32, self).setUp()
        self.overrideAttr(win32utils, 'has_ctypes', False)

class TestSetHidden(TestCaseInTempDir):

    def test_unicode_dir(self):
        if False:
            return 10
        self.requireFeature(features.UnicodeFilenameFeature)
        os.mkdir(u'ሴ')
        win32utils.set_file_attr_hidden(u'ሴ')

    def test_dot_bzr_in_unicode_dir(self):
        if False:
            print('Hello World!')
        self.requireFeature(features.UnicodeFilenameFeature)
        os.makedirs(u'ሴ\\.bzr')
        path = osutils.abspath(u'ሴ\\.bzr')
        win32utils.set_file_attr_hidden(path)

class Test_CommandLineToArgv(tests.TestCaseInTempDir):

    def assertCommandLine(self, expected, line, argv=None, single_quotes_allowed=False):
        if False:
            while True:
                i = 10
        if argv is None:
            argv = [line]
        argv = win32utils._command_line_to_argv(line, argv, single_quotes_allowed=single_quotes_allowed)
        self.assertEqual(expected, sorted(argv))

    def test_glob_paths(self):
        if False:
            print('Hello World!')
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine([u'a/b.c', u'a/c.c'], 'a/*.c')
        self.build_tree(['b/', 'b/b.c', 'b/d.c', 'b/d.h'])
        self.assertCommandLine([u'a/b.c', u'b/b.c'], '*/b.c')
        self.assertCommandLine([u'a/b.c', u'a/c.c', u'b/b.c', u'b/d.c'], '*/*.c')
        self.assertCommandLine([u'*/*.qqq'], '*/*.qqq')

    def test_quoted_globs(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine([u'a/*.c'], '"a/*.c"')
        self.assertCommandLine([u"'a/*.c'"], "'a/*.c'")
        self.assertCommandLine([u'a/*.c'], "'a/*.c'", single_quotes_allowed=True)

    def test_slashes_changed(self):
        if False:
            while True:
                i = 10
        self.assertCommandLine([u'a\\*.c'], '"a\\*.c"')
        self.assertCommandLine([u'a\\*.c'], "'a\\*.c'", single_quotes_allowed=True)
        self.assertCommandLine([u'a/*.c'], 'a\\*.c')
        self.assertCommandLine([u'a/?.c'], 'a\\?.c')
        self.assertCommandLine([u'a\\foo.c'], 'a\\foo.c')

    def test_single_quote_support(self):
        if False:
            print('Hello World!')
        self.assertCommandLine(['add', "let's-do-it.txt"], "add let's-do-it.txt", ['add', "let's-do-it.txt"])
        self.expectFailure('Using single quotes breaks trimming from argv', self.assertCommandLine, ['add', 'lets do it.txt'], "add 'lets do it.txt'", ['add', "'lets", 'do', "it.txt'"], single_quotes_allowed=True)

    def test_case_insensitive_globs(self):
        if False:
            print('Hello World!')
        if os.path.normcase('AbC') == 'AbC':
            self.skip('Test requires case insensitive globbing function')
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine([u'A/b.c'], 'A/B*')

    def test_backslashes(self):
        if False:
            return 10
        self.requireFeature(backslashdir_feature)
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine([u'a/b.c'], 'a\\b*')

    def test_with_pdb(self):
        if False:
            for i in range(10):
                print('nop')
        'Check stripping Python arguments before bzr script per lp:587868'
        self.assertCommandLine([u'rocks'], '-m pdb rocks', ['rocks'])
        self.build_tree(['d/', 'd/f1', 'd/f2'])
        self.assertCommandLine([u'rm', u'x*'], '-m pdb rm x*', ['rm', u'x*'])
        self.assertCommandLine([u'add', u'd/f1', u'd/f2'], '-m pdb add d/*', ['add', u'd/*'])

class TestGetEnvironUnicode(tests.TestCase):
    """Tests for accessing the environment via the windows wide api"""
    _test_needs_features = [CtypesFeature, features.win32_feature]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestGetEnvironUnicode, self).setUp()
        self.overrideEnv('TEST', '1')

    def test_get(self):
        if False:
            i = 10
            return i + 15
        'In the normal case behaves the same as os.environ access'
        self.assertEqual('1', win32utils.get_environ_unicode('TEST'))

    def test_unset(self):
        if False:
            i = 10
            return i + 15
        'A variable not present in the environment gives None by default'
        del os.environ['TEST']
        self.assertIs(None, win32utils.get_environ_unicode('TEST'))

    def test_unset_default(self):
        if False:
            i = 10
            return i + 15
        'A variable not present in the environment gives passed default'
        del os.environ['TEST']
        self.assertIs('a', win32utils.get_environ_unicode('TEST', 'a'))

    def test_unicode(self):
        if False:
            i = 10
            return i + 15
        'A non-ascii variable is returned as unicode'
        unicode_val = u'§'
        try:
            bytes_val = unicode_val.encode(osutils.get_user_encoding())
        except UnicodeEncodeError:
            self.skip("Couldn't encode non-ascii string to place in environ")
        os.environ['TEST'] = bytes_val
        self.assertEqual(unicode_val, win32utils.get_environ_unicode('TEST'))

    def test_long(self):
        if False:
            i = 10
            return i + 15
        'A variable bigger than heuristic buffer size is still accessible'
        big_val = 'x' * (2 << 10)
        os.environ['TEST'] = big_val
        self.assertEqual(big_val, win32utils.get_environ_unicode('TEST'))

    def test_unexpected_error(self):
        if False:
            for i in range(10):
                print('nop')
        'An error from the underlying platform function is propogated'
        ERROR_INVALID_PARAMETER = 87
        SetLastError = win32utils.ctypes.windll.kernel32.SetLastError

        def failer(*args, **kwargs):
            if False:
                print('Hello World!')
            SetLastError(ERROR_INVALID_PARAMETER)
            return 0
        self.overrideAttr(win32utils.get_environ_unicode, '_c_function', failer)
        e = self.assertRaises(WindowsError, win32utils.get_environ_unicode, 'TEST')
        self.assertEqual(e.winerror, ERROR_INVALID_PARAMETER)