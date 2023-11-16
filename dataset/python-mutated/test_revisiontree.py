"""Tests for the RevisionTree class."""
from bzrlib import errors, revision
from bzrlib.tests import TestCaseWithTransport

class TestTreeWithCommits(TestCaseWithTransport):

    def setUp(self):
        if False:
            return 10
        super(TestTreeWithCommits, self).setUp()
        self.t = self.make_branch_and_tree('.')
        self.rev_id = self.t.commit('foo', allow_pointless=True)
        self.rev_tree = self.t.branch.repository.revision_tree(self.rev_id)

    def test_empty_no_unknowns(self):
        if False:
            print('Hello World!')
        self.assertEqual([], list(self.rev_tree.unknowns()))

    def test_no_conflicts(self):
        if False:
            print('Hello World!')
        self.assertEqual([], list(self.rev_tree.conflicts()))

    def test_parents(self):
        if False:
            while True:
                i = 10
        'RevisionTree.parent_ids should match the revision graph.'
        self.assertEqual([], self.rev_tree.get_parent_ids())
        revid_2 = self.t.commit('bar', allow_pointless=True)
        self.assertEqual([self.rev_id], self.t.branch.repository.revision_tree(revid_2).get_parent_ids())
        self.assertEqual([], self.t.branch.repository.revision_tree(revision.NULL_REVISION).get_parent_ids())

    def test_empty_no_root(self):
        if False:
            for i in range(10):
                print('nop')
        null_tree = self.t.branch.repository.revision_tree(revision.NULL_REVISION)
        self.assertIs(None, null_tree.get_root_id())

    def test_get_file_revision_root(self):
        if False:
            return 10
        self.assertEqual(self.rev_id, self.rev_tree.get_file_revision(self.rev_tree.get_root_id()))

    def test_get_file_revision(self):
        if False:
            while True:
                i = 10
        self.build_tree_contents([('a', 'initial')])
        self.t.add(['a'])
        revid1 = self.t.commit('add a')
        revid2 = self.t.commit('another change', allow_pointless=True)
        tree = self.t.branch.repository.revision_tree(revid2)
        self.assertEqual(revid1, tree.get_file_revision(tree.path2id('a')))

    def test_get_file_mtime_ghost(self):
        if False:
            return 10
        file_id = iter(self.rev_tree.all_file_ids()).next()
        self.rev_tree.root_inventory[file_id].revision = 'ghostrev'
        self.assertRaises(errors.FileTimestampUnavailable, self.rev_tree.get_file_mtime, file_id)