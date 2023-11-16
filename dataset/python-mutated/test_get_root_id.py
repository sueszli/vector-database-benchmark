"""Tests for Tree.get_root_id()"""
from bzrlib.tests.per_tree import TestCaseWithTree

class TestGetRootID(TestCaseWithTree):

    def make_tree_with_default_root_id(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('tree')
        return self._convert_tree(tree)

    def make_tree_with_fixed_root_id(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('tree')
        tree.set_root_id('custom-tree-root-id')
        return self._convert_tree(tree)

    def test_get_root_id_default(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_tree_with_default_root_id()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertIsNot(None, tree.get_root_id())

    def test_get_root_id_fixed(self):
        if False:
            while True:
                i = 10
        tree = self.make_tree_with_fixed_root_id()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('custom-tree-root-id', tree.get_root_id())