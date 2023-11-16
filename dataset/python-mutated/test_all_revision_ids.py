"""Tests for all_revision_ids on a repository with external references."""
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestAllRevisionIds(TestCaseWithExternalReferenceRepository):

    def test_all_revision_ids_empty(self):
        if False:
            i = 10
            return i + 15
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        self.assertEqual(set([]), set(repo.all_revision_ids()))

    def test_all_revision_ids_from_base(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('base')
        revid = tree.commit('one')
        repo = self.make_referring('referring', tree.branch.repository)
        self.assertEqual(set([revid]), set(repo.all_revision_ids()))

    def test_all_revision_ids_from_repo(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('spare')
        revid = tree.commit('one')
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        repo.fetch(tree.branch.repository, revid)
        self.assertEqual(set([revid]), set(repo.all_revision_ids()))

    def test_all_revision_ids_from_both(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('spare')
        revid = tree.commit('one')
        base_tree = self.make_branch_and_tree('base')
        revid2 = base_tree.commit('two')
        repo = self.make_referring('referring', base_tree.branch.repository)
        repo.fetch(tree.branch.repository, revid)
        self.assertEqual(set([revid, revid2]), set(repo.all_revision_ids()))

    def test_duplicate_ids_do_not_affect_length(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('spare')
        revid = tree.commit('one')
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        repo.fetch(tree.branch.repository, revid)
        base.fetch(tree.branch.repository, revid)
        self.assertEqual(set([revid]), set(repo.all_revision_ids()))
        self.assertEqual(1, len(repo.all_revision_ids()))