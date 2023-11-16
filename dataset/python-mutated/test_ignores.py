"""Tests for handling of ignore files"""
from cStringIO import StringIO
from bzrlib import config, ignores
from bzrlib.tests import TestCase, TestCaseInTempDir, TestCaseWithTransport

class TestParseIgnoreFile(TestCase):

    def test_parse_fancy(self):
        if False:
            for i in range(10):
                print('nop')
        ignored = ignores.parse_ignore_file(StringIO('./rootdir\nrandomfile*\npath/from/ro?t\nunicodeÂµ\ndos\r\n\n#comment\n xx \n!RE:^\\.z.*\n!!./.zcompdump\n'))
        self.assertEqual(set(['./rootdir', 'randomfile*', 'path/from/ro?t', u'unicodeµ', 'dos', ' xx ', '!RE:^\\.z.*', '!!./.zcompdump']), ignored)

    def test_parse_empty(self):
        if False:
            while True:
                i = 10
        ignored = ignores.parse_ignore_file(StringIO(''))
        self.assertEqual(set([]), ignored)

    def test_parse_non_utf8(self):
        if False:
            print('Hello World!')
        'Lines with non utf 8 characters should be discarded.'
        ignored = ignores.parse_ignore_file(StringIO('utf8filename_a\ninvalid utf8\x80\nutf8filename_b\n'))
        self.assertEqual(set(['utf8filename_a', 'utf8filename_b']), ignored)

class TestUserIgnores(TestCaseInTempDir):

    def test_create_if_missing(self):
        if False:
            print('Hello World!')
        ignore_path = config.user_ignore_config_filename()
        self.assertPathDoesNotExist(ignore_path)
        user_ignores = ignores.get_user_ignores()
        self.assertEqual(set(ignores.USER_DEFAULTS), user_ignores)
        self.assertPathExists(ignore_path)
        f = open(ignore_path, 'rb')
        try:
            entries = ignores.parse_ignore_file(f)
        finally:
            f.close()
        self.assertEqual(set(ignores.USER_DEFAULTS), entries)

    def test_use_existing(self):
        if False:
            while True:
                i = 10
        patterns = ['*.o', '*.py[co]', u'å*']
        ignores._set_user_ignores(patterns)
        user_ignores = ignores.get_user_ignores()
        self.assertEqual(set(patterns), user_ignores)

    def test_use_empty(self):
        if False:
            print('Hello World!')
        ignores._set_user_ignores([])
        ignore_path = config.user_ignore_config_filename()
        self.check_file_contents(ignore_path, '')
        self.assertEqual(set([]), ignores.get_user_ignores())

    def test_set(self):
        if False:
            for i in range(10):
                print('nop')
        patterns = ['*.py[co]', '*.py[oc]']
        ignores._set_user_ignores(patterns)
        self.assertEqual(set(patterns), ignores.get_user_ignores())
        patterns = ['vim', '*.swp']
        ignores._set_user_ignores(patterns)
        self.assertEqual(set(patterns), ignores.get_user_ignores())

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that adding will not duplicate ignores'
        ignores._set_user_ignores([])
        patterns = ['foo', './bar', u'båz']
        added = ignores.add_unique_user_ignores(patterns)
        self.assertEqual(patterns, added)
        self.assertEqual(set(patterns), ignores.get_user_ignores())

    def test_add_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that adding a directory will strip any trailing slash'
        ignores._set_user_ignores([])
        in_patterns = ['foo/', 'bar/', 'baz\\']
        added = ignores.add_unique_user_ignores(in_patterns)
        out_patterns = [x.rstrip('/\\') for x in in_patterns]
        self.assertEqual(out_patterns, added)
        self.assertEqual(set(out_patterns), ignores.get_user_ignores())

    def test_add_unique(self):
        if False:
            return 10
        'Test that adding will not duplicate ignores'
        ignores._set_user_ignores(['foo', './bar', u'båz', 'dir1/', 'dir3\\'])
        added = ignores.add_unique_user_ignores(['xxx', './bar', 'xxx', 'dir1/', 'dir2/', 'dir3\\'])
        self.assertEqual(['xxx', 'dir2'], added)
        self.assertEqual(set(['foo', './bar', u'båz', 'xxx', 'dir1', 'dir2', 'dir3']), ignores.get_user_ignores())

class TestRuntimeIgnores(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestRuntimeIgnores, self).setUp()
        self.overrideAttr(ignores, '_runtime_ignores', set())

    def test_add(self):
        if False:
            while True:
                i = 10
        'Test that we can add an entry to the list.'
        self.assertEqual(set(), ignores.get_runtime_ignores())
        ignores.add_runtime_ignores(['foo'])
        self.assertEqual(set(['foo']), ignores.get_runtime_ignores())

    def test_add_duplicate(self):
        if False:
            for i in range(10):
                print('nop')
        "Adding the same ignore twice shouldn't add a new entry."
        ignores.add_runtime_ignores(['foo', 'bar'])
        self.assertEqual(set(['foo', 'bar']), ignores.get_runtime_ignores())
        ignores.add_runtime_ignores(['bar'])
        self.assertEqual(set(['foo', 'bar']), ignores.get_runtime_ignores())

class TestTreeIgnores(TestCaseWithTransport):

    def assertPatternsEquals(self, patterns):
        if False:
            print('Hello World!')
        contents = open('.bzrignore', 'rU').read().strip().split('\n')
        self.assertEqual(sorted(patterns), sorted(contents))

    def test_new_file(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        ignores.tree_ignores_add_patterns(tree, ['myentry'])
        self.assertTrue(tree.has_filename('.bzrignore'))
        self.assertPatternsEquals(['myentry'])

    def test_add_to_existing(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', 'myentry1\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2', 'foo'])
        self.assertPatternsEquals(['myentry1', 'myentry2', 'foo'])

    def test_adds_ending_newline(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', 'myentry1')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2'])
        self.assertPatternsEquals(['myentry1', 'myentry2'])
        text = open('.bzrignore', 'r').read()
        self.assertTrue(text.endswith('\r\n') or text.endswith('\n') or text.endswith('\r'))

    def test_does_not_add_dupe(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', 'myentry\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry'])
        self.assertPatternsEquals(['myentry'])

    def test_non_ascii(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', u'myentryሴ\n'.encode('utf-8'))])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, [u'myentry噸'])
        self.assertPatternsEquals([u'myentryሴ'.encode('utf-8'), u'myentry噸'.encode('utf-8')])

    def test_crlf(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', 'myentry1\r\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2', 'foo'])
        self.assertEqual(open('.bzrignore', 'rb').read(), 'myentry1\r\nmyentry2\r\nfoo\r\n')
        self.assertPatternsEquals(['myentry1', 'myentry2', 'foo'])