from bzrlib import inventory
from bzrlib.tests import TestNotApplicable
from bzrlib.transform import TreeTransform
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestNestedSupport(TestCaseWithWorkingTree):

    def make_branch_and_tree(self, path):
        if False:
            while True:
                i = 10
        tree = TestCaseWithWorkingTree.make_branch_and_tree(self, path)
        if not tree.supports_tree_reference():
            raise TestNotApplicable('Tree references not supported')
        return tree

    def test_set_get_tree_reference(self):
        if False:
            while True:
                i = 10
        'This tests that setting a tree reference is persistent.'
        tree = self.make_branch_and_tree('.')
        transform = TreeTransform(tree)
        trans_id = transform.new_directory('reference', transform.root, 'subtree-id')
        transform.set_tree_reference('subtree-revision', trans_id)
        transform.apply()
        tree = tree.bzrdir.open_workingtree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('subtree-revision', tree.root_inventory['subtree-id'].reference_revision)

    def test_extract_while_locked(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['subtree/'])
        tree.add(['subtree'], ['subtree-id'])
        subtree = tree.extract('subtree-id')

    def prepare_with_subtree(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        subtree = self.make_branch_and_tree('subtree')
        tree.add(['subtree'], ['subtree-id'])
        return tree

    def test_kind_does_not_autodetect_subtree(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.prepare_with_subtree()
        self.assertEqual('directory', tree.kind('subtree-id'))

    def test_comparison_data_does_not_autodetect_subtree(self):
        if False:
            i = 10
            return i + 15
        tree = self.prepare_with_subtree()
        ie = inventory.InventoryDirectory('subtree-id', 'subtree', tree.path2id(''))
        self.assertEqual('directory', tree._comparison_data(ie, 'subtree')[0])

    def test_inventory_does_not_autodetect_subtree(self):
        if False:
            i = 10
            return i + 15
        tree = self.prepare_with_subtree()
        self.assertEqual('directory', tree.kind('subtree-id'))

    def test_iter_entries_by_dir_autodetects_subtree(self):
        if False:
            return 10
        tree = self.prepare_with_subtree()
        (path, ie) = tree.iter_entries_by_dir(['subtree-id']).next()
        self.assertEqual('tree-reference', ie.kind)