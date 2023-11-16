"""Test the find_text_key_references API."""
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

class TestFindTextKeyReferences(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        repo = self.make_repository('.')
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({}, repo.find_text_key_references())