"""Tests for MutableTree.

Most functionality of MutableTree is tested as part of WorkingTree.
"""
from bzrlib import mutabletree, tests

class TestHooks(tests.TestCase):

    def test_constructor(self):
        if False:
            while True:
                i = 10
        'Check that creating a MutableTreeHooks instance has the right\n        defaults.'
        hooks = mutabletree.MutableTreeHooks()
        self.assertTrue('start_commit' in hooks, 'start_commit not in %s' % hooks)
        self.assertTrue('post_commit' in hooks, 'post_commit not in %s' % hooks)

    def test_installed_hooks_are_MutableTreeHooks(self):
        if False:
            for i in range(10):
                print('nop')
        'The installed hooks object should be a MutableTreeHooks.'
        self.assertIsInstance(self._preserved_hooks[mutabletree.MutableTree][1], mutabletree.MutableTreeHooks)

class TestHasChanges(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            return 10
        super(TestHasChanges, self).setUp()
        self.tree = self.make_branch_and_tree('tree')

    def test_with_uncommitted_changes(self):
        if False:
            return 10
        self.build_tree(['tree/file'])
        self.tree.add('file')
        self.assertTrue(self.tree.has_changes())

    def test_with_pending_merges(self):
        if False:
            return 10
        self.tree.commit('first commit')
        other_tree = self.tree.bzrdir.sprout('other').open_workingtree()
        other_tree.commit('mergeable commit')
        self.tree.merge_from_branch(other_tree.branch)
        self.assertTrue(self.tree.has_changes())