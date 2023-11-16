"""Tests for implementations of Repository.has_revisions."""
from bzrlib.revision import NULL_REVISION
from bzrlib.tests.per_repository import TestCaseWithRepository

class TestHasRevisions(TestCaseWithRepository):

    def test_empty_list(self):
        if False:
            print('Hello World!')
        repo = self.make_repository('.')
        self.assertEqual(set(), repo.has_revisions([]))

    def test_superset(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        repo = tree.branch.repository
        rev1 = tree.commit('1')
        rev2 = tree.commit('2')
        rev3 = tree.commit('3')
        self.assertEqual(set([rev1, rev3]), repo.has_revisions([rev1, rev3, 'foobar:']))

    def test_NULL(self):
        if False:
            while True:
                i = 10
        repo = self.make_repository('.')
        self.assertEqual(set([NULL_REVISION]), repo.has_revisions([NULL_REVISION]))