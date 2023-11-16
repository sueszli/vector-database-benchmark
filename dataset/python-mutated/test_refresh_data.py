"""Tests for VersionedFileRepository.refresh_data."""
from bzrlib import repository
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

class TestRefreshData(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def fetch_new_revision_into_concurrent_instance(self, repo, token):
        if False:
            i = 10
            return i + 15
        "Create a new revision (revid 'new-rev') and fetch it into a\n        concurrent instance of repo.\n        "
        source = self.make_branch_and_memory_tree('source')
        source.lock_write()
        self.addCleanup(source.unlock)
        source.add([''], ['root-id'])
        revid = source.commit('foo', rev_id='new-rev')
        repo.all_revision_ids()
        repo.revisions.keys()
        repo.inventories.keys()
        server_repo = repo.bzrdir.open_repository()
        try:
            server_repo.lock_write(token)
        except errors.TokenLockingNotSupported:
            raise TestSkipped('Cannot concurrently insert into repo format %r' % self.repository_format)
        try:
            server_repo.fetch(source.branch.repository, revid)
        finally:
            server_repo.unlock()

    def test_refresh_data_after_fetch_new_data_visible_in_write_group(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_memory_tree('target')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ['root-id'])
        tree.commit('foo', rev_id='commit-in-target')
        repo = tree.branch.repository
        token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        self.addCleanup(repo.abort_write_group)
        self.fetch_new_revision_into_concurrent_instance(repo, token)
        try:
            repo.refresh_data()
        except repository.IsInWriteGroupError:
            pass
        else:
            self.assertEqual(['commit-in-target', 'new-rev'], sorted(repo.all_revision_ids()))

    def test_refresh_data_after_fetch_new_data_visible(self):
        if False:
            for i in range(10):
                print('nop')
        repo = self.make_repository('target')
        token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        self.fetch_new_revision_into_concurrent_instance(repo, token)
        repo.refresh_data()
        self.assertNotEqual({}, repo.get_graph().get_parent_map(['new-rev']))