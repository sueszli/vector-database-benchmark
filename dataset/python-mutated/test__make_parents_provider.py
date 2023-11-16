"""Tests for _make_parents_provider on stacked repositories."""
from bzrlib.tests.per_repository import TestCaseWithRepository

class Test_MakeParentsProvider(TestCaseWithRepository):

    def test_add_fallback_after_make_pp(self):
        if False:
            i = 10
            return i + 15
        'Fallbacks added after _make_parents_provider are used by that\n        provider.\n        '
        referring_repo = self.make_repository('repo')
        pp = referring_repo._make_parents_provider()
        self.addCleanup(referring_repo.lock_read().unlock)
        self.assertEqual({}, pp.get_parent_map(['revid2']))
        wt_a = self.make_branch_and_tree('fallback')
        wt_a.commit('first commit', rev_id='revid1')
        wt_a.commit('second commit', rev_id='revid2')
        fallback_repo = wt_a.branch.repository
        referring_repo.add_fallback_repository(fallback_repo)
        self.assertEqual(('revid1',), pp.get_parent_map(['revid2'])['revid2'])