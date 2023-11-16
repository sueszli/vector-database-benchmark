"""Tests for Branch.revision_id_to_revno()"""
from bzrlib import errors
from bzrlib.tests import TestNotApplicable
from bzrlib.tests.per_branch import TestCaseWithBranch

class TestRevisionIdToRevno(TestCaseWithBranch):

    def test_simple_revno(self):
        if False:
            while True:
                i = 10
        tree = self.create_tree_with_merge()
        the_branch = tree.branch
        self.assertEqual(0, the_branch.revision_id_to_revno('null:'))
        self.assertEqual(1, the_branch.revision_id_to_revno('rev-1'))
        self.assertEqual(2, the_branch.revision_id_to_revno('rev-2'))
        self.assertEqual(3, the_branch.revision_id_to_revno('rev-3'))
        self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, 'rev-none')
        self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, 'rev-1.1.1')

    def test_mainline_ghost(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree1')
        if not tree.branch.repository._format.supports_ghosts:
            raise TestNotApplicable('repository format does not support ghosts')
        tree.set_parent_ids(['spooky'], allow_leftmost_as_ghost=True)
        tree.add('')
        tree.commit('msg1', rev_id='rev1')
        tree.commit('msg2', rev_id='rev2')
        self.assertRaises((errors.NoSuchRevision, errors.GhostRevisionsHaveNoRevno), tree.branch.revision_id_to_revno, 'unknown')
        self.assertEqual(1, tree.branch.revision_id_to_revno('rev1'))
        self.assertEqual(2, tree.branch.revision_id_to_revno('rev2'))

    def test_empty(self):
        if False:
            print('Hello World!')
        branch = self.make_branch('.')
        self.assertRaises(errors.NoSuchRevision, branch.revision_id_to_revno, 'unknown')
        self.assertEqual(0, branch.revision_id_to_revno('null:'))