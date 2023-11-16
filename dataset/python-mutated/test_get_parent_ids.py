"""Tests for interface conformance of 'workingtree.get_parent_ids'"""
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestGetParentIds(TestCaseWithWorkingTree):

    def test_get_parent_ids(self):
        if False:
            i = 10
            return i + 15
        t = self.make_branch_and_tree('t1')
        self.assertEqual([], t.get_parent_ids())
        rev1_id = t.commit('foo', allow_pointless=True)
        self.assertEqual([rev1_id], t.get_parent_ids())
        t2 = t.bzrdir.sprout('t2').open_workingtree()
        rev2_id = t2.commit('foo', allow_pointless=True)
        self.assertEqual([rev2_id], t2.get_parent_ids())
        t.merge_from_branch(t2.branch)
        self.assertEqual([rev1_id, rev2_id], t.get_parent_ids())
        for parent_id in t.get_parent_ids():
            self.assertIsInstance(parent_id, str)

    def test_pending_merges(self):
        if False:
            return 10
        'Test the correspondence between set pending merges and get_parent_ids.'
        wt = self.make_branch_and_tree('.')
        self.assertEqual([], wt.get_parent_ids())
        wt.add_pending_merge('foo@azkhazan-123123-abcabc')
        self.assertEqual(['foo@azkhazan-123123-abcabc'], wt.get_parent_ids())
        wt.add_pending_merge('foo@azkhazan-123123-abcabc')
        self.assertEqual(['foo@azkhazan-123123-abcabc'], wt.get_parent_ids())
        wt.add_pending_merge('wibble@fofof--20050401--1928390812')
        self.assertEqual(['foo@azkhazan-123123-abcabc', 'wibble@fofof--20050401--1928390812'], wt.get_parent_ids())