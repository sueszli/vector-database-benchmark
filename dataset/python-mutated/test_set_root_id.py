"""Tests for WorkingTree.set_root_id"""
import sys
from bzrlib import errors
from bzrlib.tests import TestSkipped
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestSetRootId(TestCaseWithWorkingTree):

    def test_set_and_read_unicode(self):
        if False:
            print('Hello World!')
        if sys.platform == 'win32':
            raise TestSkipped("don't use oslocks on win32 in unix manner")
        self.thisFailsStrictLockCheck()
        tree = self.make_branch_and_tree('a-tree')
        root_id = u'ån-id'.encode('utf8')
        tree.lock_write()
        try:
            old_id = tree.get_root_id()
            tree.set_root_id(root_id)
            self.assertEqual(root_id, tree.get_root_id())
            reference_tree = tree.bzrdir.open_workingtree()
            self.assertEqual(old_id, reference_tree.get_root_id())
        finally:
            tree.unlock()
        self.assertEqual(root_id, tree.get_root_id())
        tree = tree.bzrdir.open_workingtree()
        self.assertEqual(root_id, tree.get_root_id())
        tree._validate()

    def test_set_root_id(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        orig_root_id = tree.get_root_id()
        self.assertNotEqual('custom-root-id', orig_root_id)
        self.assertEqual('', tree.id2path(orig_root_id))
        self.assertRaises(errors.NoSuchId, tree.id2path, 'custom-root-id')
        tree.set_root_id('custom-root-id')
        self.assertEqual('custom-root-id', tree.get_root_id())
        self.assertEqual('custom-root-id', tree.path2id(''))
        self.assertEqual('', tree.id2path('custom-root-id'))
        self.assertRaises(errors.NoSuchId, tree.id2path, orig_root_id)
        tree._validate()