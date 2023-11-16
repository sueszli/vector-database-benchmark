"""Tests for WorkingTree.read_working_inventory."""
from bzrlib import errors, inventory
from bzrlib.tests import TestNotApplicable
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree
from bzrlib.workingtree import InventoryWorkingTree

class TestReadWorkingInventory(TestCaseWithWorkingTree):

    def test_trivial_read(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('t1')
        if not isinstance(tree, InventoryWorkingTree):
            raise TestNotApplicable('read_working_inventory not usable on non-inventory working trees')
        tree.lock_read()
        self.assertIsInstance(tree.read_working_inventory(), inventory.Inventory)
        tree.unlock()

    def test_read_after_inventory_modification(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        if not isinstance(tree, InventoryWorkingTree):
            raise TestNotApplicable('read_working_inventory not usable on non-inventory working trees')
        tree.lock_write()
        try:
            tree.set_root_id('new-root')
            try:
                tree.read_working_inventory()
            except errors.InventoryModified:
                pass
            else:
                self.assertEqual('new-root', tree.path2id(''))
        finally:
            tree.unlock()