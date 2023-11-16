"""Tests for add_inventory on a repository with external references."""
from bzrlib import errors
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestAddInventory(TestCaseWithExternalReferenceRepository):

    def test_add_inventory_goes_to_repo(self):
        if False:
            print('Hello World!')
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
                repo.add_inventory(revid, inv, [])
            except:
                repo.abort_write_group()
                raise
            else:
                repo.commit_write_group()
        finally:
            repo.unlock()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        inv2 = repo.get_inventory(revid)
        content1 = dict(((file_id, inv[file_id]) for file_id in inv))
        content2 = dict(((file_id, inv[file_id]) for file_id in inv2))
        self.assertEqual(content1, content2)
        self.assertRaises(errors.NoSuchRevision, base.get_inventory, revid)