"""Black-box tests for 'bzr unknowns', which shows unknown files."""
from bzrlib.tests import TestCaseWithTransport

class TestUnknowns(TestCaseWithTransport):

    def test_unknowns(self):
        if False:
            while True:
                i = 10
        "Test that 'unknown' command reports unknown files"
        tree = self.make_branch_and_tree('.')
        self.assertEqual(self.run_bzr('unknowns')[0], '')
        self.build_tree_contents([('a', 'contents of a\n')])
        self.assertEqual(self.run_bzr('unknowns')[0], 'a\n')
        self.build_tree(['b', 'c', 'd e'])
        self.assertEqual(self.run_bzr('unknowns')[0], 'a\nb\nc\n"d e"\n')
        tree.add(['a', 'd e'])
        self.assertEqual(self.run_bzr('unknowns')[0], 'b\nc\n')
        tree.add(['b', 'c'])
        self.assertEqual(self.run_bzr('unknowns')[0], '')

    def test_unknowns_directory(self):
        if False:
            return 10
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        (out, err) = self.run_bzr(['unknowns', '--directory=a'])
        self.assertEqual('README\n', out)