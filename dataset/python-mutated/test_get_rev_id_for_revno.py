"""Tests for get_rev_id_for_revno on a repository with external references."""
from bzrlib import errors
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestGetRevIdForRevno(TestCaseWithExternalReferenceRepository):

    def test_uses_fallback(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('base')
        base = tree.branch.repository
        revid = tree.commit('one')
        revid2 = tree.commit('two')
        spare_tree = tree.bzrdir.sprout('spare').open_workingtree()
        revid3 = spare_tree.commit('three')
        branch = spare_tree.branch.create_clone_on_transport(self.get_transport('referring'), stacked_on=tree.branch.base)
        repo = branch.repository
        self.assertEqual(set([revid3]), set(repo.bzrdir.open_repository().all_revision_ids()))
        self.assertEqual(set([revid2, revid]), set(base.bzrdir.open_repository().all_revision_ids()))
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual((True, revid), repo.get_rev_id_for_revno(1, (3, revid3)))