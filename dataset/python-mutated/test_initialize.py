"""Tests for initializing a repository with external references."""
from bzrlib import errors, tests
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestInitialize(TestCaseWithExternalReferenceRepository):

    def initialize_and_check_on_transport(self, base, trans):
        if False:
            for i in range(10):
                print('nop')
        network_name = base.repository._format.network_name()
        result = self.bzrdir_format.initialize_on_transport_ex(trans, use_existing_dir=False, create_prefix=False, stacked_on='../base', stack_on_pwd=base.base, repo_format_name=network_name)
        (result_repo, a_bzrdir, require_stacking, repo_policy) = result
        self.addCleanup(result_repo.unlock)
        self.assertEqual(1, len(result_repo._fallback_repositories))
        return result_repo

    def test_initialize_on_transport_ex(self):
        if False:
            return 10
        base = self.make_branch('base')
        trans = self.get_transport('stacked')
        repo = self.initialize_and_check_on_transport(base, trans)
        self.assertEqual(base.repository._format.network_name(), repo._format.network_name())

    def test_remote_initialize_on_transport_ex(self):
        if False:
            return 10
        base = self.make_branch('base')
        trans = self.make_smart_server('stacked')
        repo = self.initialize_and_check_on_transport(base, trans)
        network_name = base.repository._format.network_name()
        self.assertEqual(network_name, repo._format.network_name())