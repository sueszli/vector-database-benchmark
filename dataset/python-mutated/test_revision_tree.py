"""Tests for Tree.revision_tree."""
from bzrlib import errors, tests
from bzrlib.tests import per_tree

class TestRevisionTree(per_tree.TestCaseWithTree):

    def create_tree_no_parents_no_content(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        return self.get_tree_no_parents_no_content(tree)

    def test_get_random_tree_raises(self):
        if False:
            i = 10
            return i + 15
        test_tree = self.create_tree_no_parents_no_content()
        self.assertRaises(errors.NoSuchRevision, test_tree.revision_tree, 'this-should-not-exist')