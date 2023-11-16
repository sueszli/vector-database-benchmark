"""Tests for Branch.dotted_revno_to_revision_id()"""
from bzrlib import errors
from bzrlib.tests.per_branch import TestCaseWithBranch

class TestDottedRevnoToRevisionId(TestCaseWithBranch):

    def test_lookup_revision_id_by_dotted(self):
        if False:
            while True:
                i = 10
        tree = self.create_tree_with_merge()
        the_branch = tree.branch
        the_branch.lock_read()
        self.addCleanup(the_branch.unlock)
        self.assertEqual('null:', the_branch.dotted_revno_to_revision_id((0,)))
        self.assertEqual('rev-1', the_branch.dotted_revno_to_revision_id((1,)))
        self.assertEqual('rev-2', the_branch.dotted_revno_to_revision_id((2,)))
        self.assertEqual('rev-3', the_branch.dotted_revno_to_revision_id((3,)))
        self.assertEqual('rev-1.1.1', the_branch.dotted_revno_to_revision_id((1, 1, 1)))
        self.assertRaises(errors.NoSuchRevision, the_branch.dotted_revno_to_revision_id, (1, 0, 2))
        self.assertEqual(None, the_branch._partial_revision_id_to_revno_cache.get('rev-1'))
        self.assertEqual('rev-1', the_branch.dotted_revno_to_revision_id((1,), _cache_reverse=True))
        self.assertEqual((1,), the_branch._partial_revision_id_to_revno_cache.get('rev-1'))