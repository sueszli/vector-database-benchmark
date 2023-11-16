"""Tests for the InterTree.file_content_matches() function."""
from bzrlib.tests.per_intertree import TestCaseWithTwoTrees

class TestFileContentMatches(TestCaseWithTwoTrees):

    def test_same_contents_and_verifier(self):
        if False:
            return 10
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        self.build_tree_contents([('1/file', 'apples'), ('2/file', 'apples')])
        tree1.add('file', 'file-id-1')
        tree2.add('file', 'file-id-2')
        (tree1, tree2) = self.mutable_trees_to_test_trees(self, tree1, tree2)
        inter = self.intertree_class(tree1, tree2)
        self.assertTrue(inter.file_content_matches('file-id-1', 'file-id-2'))

    def test_different_contents_and_same_verifier(self):
        if False:
            i = 10
            return i + 15
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        self.build_tree_contents([('1/file', 'apples'), ('2/file', 'oranges')])
        tree1.add('file', 'file-id-1')
        tree2.add('file', 'file-id-2')
        (tree1, tree2) = self.mutable_trees_to_test_trees(self, tree1, tree2)
        inter = self.intertree_class(tree1, tree2)
        self.assertFalse(inter.file_content_matches('file-id-1', 'file-id-2'))