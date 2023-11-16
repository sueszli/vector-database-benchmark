from bzrlib import config, ignores
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestIsIgnored(TestCaseWithWorkingTree):

    def _set_user_ignore_content(self, ignores):
        if False:
            for i in range(10):
                print('nop')
        'Create user ignore file and set its content to ignores.'
        config.ensure_config_dir_exists()
        user_ignore_file = config.user_ignore_config_filename()
        f = open(user_ignore_file, 'wb')
        try:
            f.write(ignores)
        finally:
            f.close()

    def test_is_ignored(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', './rootdir\nrandomfile*\n*bar\n!bazbar\n?foo\n*.~*\ndir1/*f1\ndir1/?f2\nRE:dir2/.*\\.wombat\npath/from/ro?t\n**/piffle.py\n!b/piffle.py\nunicodeÂµ\ndos\r\n\n#comment\n xx \n')])
        self._set_user_ignore_content('')
        self.assertEqual('./rootdir', tree.is_ignored('rootdir'))
        self.assertEqual(None, tree.is_ignored('foo/rootdir'))
        self.assertEqual(None, tree.is_ignored('rootdirtrailer'))
        self.assertEqual('randomfile*', tree.is_ignored('randomfile'))
        self.assertEqual('randomfile*', tree.is_ignored('randomfiles'))
        self.assertEqual('randomfile*', tree.is_ignored('foo/randomfiles'))
        self.assertEqual(None, tree.is_ignored('randomfil'))
        self.assertEqual(None, tree.is_ignored('foo/randomfil'))
        self.assertEqual('path/from/ro?t', tree.is_ignored('path/from/root'))
        self.assertEqual('path/from/ro?t', tree.is_ignored('path/from/roat'))
        self.assertEqual(None, tree.is_ignored('roat'))
        self.assertEqual('**/piffle.py', tree.is_ignored('piffle.py'))
        self.assertEqual('**/piffle.py', tree.is_ignored('a/piffle.py'))
        self.assertEqual(None, tree.is_ignored('b/piffle.py'))
        self.assertEqual('**/piffle.py', tree.is_ignored('foo/bar/piffle.py'))
        self.assertEqual(None, tree.is_ignored('p/iffle.py'))
        self.assertEqual(u'unicodeµ', tree.is_ignored(u'unicodeµ'))
        self.assertEqual(u'unicodeµ', tree.is_ignored(u'subdir/unicodeµ'))
        self.assertEqual(None, tree.is_ignored(u'unicodeå'))
        self.assertEqual(None, tree.is_ignored(u'unicode'))
        self.assertEqual(None, tree.is_ignored(u'µ'))
        self.assertEqual('dos', tree.is_ignored('dos'))
        self.assertEqual(None, tree.is_ignored('dosfoo'))
        self.assertEqual('*bar', tree.is_ignored('foobar'))
        self.assertEqual('*bar', tree.is_ignored('foo\\nbar'))
        self.assertEqual('*bar', tree.is_ignored('bar'))
        self.assertEqual('*bar', tree.is_ignored('.bar'))
        self.assertEqual(None, tree.is_ignored('bazbar'))
        self.assertEqual('?foo', tree.is_ignored('afoo'))
        self.assertEqual('?foo', tree.is_ignored('.foo'))
        self.assertEqual('*.~*', tree.is_ignored('blah.py.~1~'))
        self.assertEqual('dir1/*f1', tree.is_ignored('dir1/foof1'))
        self.assertEqual('dir1/*f1', tree.is_ignored('dir1/f1'))
        self.assertEqual('dir1/*f1', tree.is_ignored('dir1/.f1'))
        self.assertEqual('dir1/?f2', tree.is_ignored('dir1/ff2'))
        self.assertEqual('dir1/?f2', tree.is_ignored('dir1/.f2'))
        self.assertEqual('RE:dir2/.*\\.wombat', tree.is_ignored('dir2/foo.wombat'))
        self.assertEqual(None, tree.is_ignored('dir2/foo'))
        self.assertEqual(None, tree.is_ignored(''))
        self.assertEqual(None, tree.is_ignored('test/'))
        self.assertEqual(None, tree.is_ignored('#comment'))
        self.assertEqual(' xx ', tree.is_ignored(' xx '))
        self.assertEqual(' xx ', tree.is_ignored('subdir/ xx '))
        self.assertEqual(None, tree.is_ignored('xx'))
        self.assertEqual(None, tree.is_ignored('xx '))
        self.assertEqual(None, tree.is_ignored(' xx'))
        self.assertEqual(None, tree.is_ignored('subdir/xx '))

    def test_global_ignored(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        config.ensure_config_dir_exists()
        user_ignore_file = config.user_ignore_config_filename()
        self._set_user_ignore_content('*.py[co]\n./.shelf\n# comment line\n\n\r\n * \ncrlf\r\n*Ã¥*\n')
        self.assertEqual('./.shelf', tree.is_ignored('.shelf'))
        self.assertEqual(None, tree.is_ignored('foo/.shelf'))
        self.assertEqual('*.py[co]', tree.is_ignored('foo.pyc'))
        self.assertEqual('*.py[co]', tree.is_ignored('foo.pyo'))
        self.assertEqual(None, tree.is_ignored('foo.py'))
        self.assertEqual('*.py[co]', tree.is_ignored('bar/foo.pyc'))
        self.assertEqual('*.py[co]', tree.is_ignored('bar/foo.pyo'))
        self.assertEqual(None, tree.is_ignored('bar/foo.py'))
        self.assertEqual(u'*å*', tree.is_ignored(u'bågfors'))
        self.assertEqual(u'*å*', tree.is_ignored(u'ågfors'))
        self.assertEqual(u'*å*', tree.is_ignored(u'å'))
        self.assertEqual(u'*å*', tree.is_ignored(u'bå'))
        self.assertEqual(u'*å*', tree.is_ignored(u'b/å'))
        self.assertEqual(' * ', tree.is_ignored(' bbb '))
        self.assertEqual(' * ', tree.is_ignored('subdir/ bbb '))
        self.assertEqual(None, tree.is_ignored('bbb '))
        self.assertEqual(None, tree.is_ignored(' bbb'))
        self.assertEqual('crlf', tree.is_ignored('crlf'))
        self.assertEqual('crlf', tree.is_ignored('subdir/crlf'))
        self.assertEqual(None, tree.is_ignored('# comment line'))
        self.assertEqual(None, tree.is_ignored(''))
        self.assertEqual(None, tree.is_ignored('baz/'))

    def test_mixed_is_ignored(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        ignores._set_user_ignores(['*.py[co]', './.shelf'])
        self.build_tree_contents([('.bzrignore', './rootdir\n*.swp\n')])
        self.assertEqual('*.py[co]', tree.is_ignored('foo.pyc'))
        self.assertEqual('./.shelf', tree.is_ignored('.shelf'))
        self.assertEqual('./rootdir', tree.is_ignored('rootdir'))
        self.assertEqual('*.swp', tree.is_ignored('foo.py.swp'))
        self.assertEqual('*.swp', tree.is_ignored('.foo.py.swp'))
        self.assertEqual(None, tree.is_ignored('.foo.py.swo'))

    def test_runtime_ignores(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', '')])
        ignores._set_user_ignores([])
        orig_runtime = ignores._runtime_ignores
        try:
            ignores._runtime_ignores = set()
            self.assertEqual(None, tree.is_ignored('foobar.py'))
            tree._flush_ignore_list_cache()
            ignores.add_runtime_ignores(['./foobar.py'])
            self.assertEqual(set(['./foobar.py']), ignores.get_runtime_ignores())
            self.assertEqual('./foobar.py', tree.is_ignored('foobar.py'))
        finally:
            ignores._runtime_ignores = orig_runtime

    def test_ignore_caching(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['ignoreme'])
        self.assertEqual(None, tree.is_ignored('ignoreme'))
        tree.unknowns()
        self.build_tree_contents([('.bzrignore', 'ignoreme')])
        self.assertEqual('ignoreme', tree.is_ignored('ignoreme'))