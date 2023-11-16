import os
from bzrlib.tests import TestCaseWithTransport

class TestFindMergeBase(TestCaseWithTransport):

    def test_find_merge_base(self):
        if False:
            for i in range(10):
                print('nop')
        a_tree = self.make_branch_and_tree('a')
        a_tree.commit(message='foo', allow_pointless=True)
        b_tree = a_tree.bzrdir.sprout('b').open_workingtree()
        q = self.run_bzr('find-merge-base a b')[0]
        a_tree.commit(message='bar', allow_pointless=True)
        b_tree.commit(message='baz', allow_pointless=True)
        r = self.run_bzr('find-merge-base b a')[0]
        self.assertEqual(q, r)

    def test_find_null_merge_base(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('foo')
        tree.commit('message')
        tree2 = self.make_branch_and_tree('bar')
        r = self.run_bzr('find-merge-base foo bar')[0]
        self.assertEqual('merge base is revision null:\n', r)