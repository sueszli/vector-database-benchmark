"""Tests for add_revision on a repository with external references."""
from bzrlib import errors
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestAddRevision(TestCaseWithExternalReferenceRepository):

    def test_add_revision_goes_to_repo(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('sample')
        revid = tree.commit('one')
        inv = tree.branch.repository.get_inventory(revid)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        rev = tree.branch.repository.get_revision(revid)
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                rev = tree.branch.repository.get_revision(revid)
                repo.texts.add_lines((inv.root.file_id, revid), [], [])
                repo.add_revision(revid, rev, inv=inv)
            except:
                repo.abort_write_group()
                raise
            else:
                repo.commit_write_group()
        finally:
            repo.unlock()
        rev2 = repo.get_revision(revid)
        self.assertEqual(rev, rev2)
        self.assertRaises(errors.NoSuchRevision, base.get_revision, revid)