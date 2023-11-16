"""Tests of the 'bzr ignored' command."""
from bzrlib.tests import TestCaseWithTransport

class TestIgnored(TestCaseWithTransport):

    def test_ignored_added_file(self):
        if False:
            while True:
                i = 10
        "'bzr ignored' should not list versioned files."
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo.pyc'])
        self.build_tree_contents([('.bzrignore', 'foo.pyc')])
        self.assertTrue(tree.is_ignored('foo.pyc'))
        tree.add('foo.pyc')
        (out, err) = self.run_bzr('ignored')
        self.assertEqual('', out)
        self.assertEqual('', err)

    def test_ignored_directory(self):
        if False:
            while True:
                i = 10
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree_contents([('a/README', 'contents'), ('a/.bzrignore', 'README')])
        (out, err) = self.run_bzr(['ignored', '--directory=a'])
        self.assertStartsWith(out, 'README')