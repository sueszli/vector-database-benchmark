"""Tests for add_signature_text on a repository with external references."""
from bzrlib import errors
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestAddSignatureText(TestCaseWithExternalReferenceRepository):

    def test_add_signature_text_goes_to_repo(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('sample')
        revid = tree.commit('one')
        inv = tree.branch.repository.get_inventory(revid)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                rev = tree.branch.repository.get_revision(revid)
                repo.texts.add_lines((inv.root.file_id, revid), [], [])
                repo.add_revision(revid, rev, inv=inv)
                repo.add_signature_text(revid, 'text')
                repo.commit_write_group()
            except:
                repo.abort_write_group()
                raise
        finally:
            repo.unlock()
        repo.get_signature_text(revid)
        self.assertRaises(errors.NoSuchRevision, base.get_signature_text, revid)