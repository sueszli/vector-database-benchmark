"""Tests for Branch.revision_id_to_dotted_revno()"""
from bzrlib import errors
from bzrlib.tests.per_branch import TestCaseWithBranch

class TestRevisionIdToDottedRevno(TestCaseWithBranch):

    def test_lookup_dotted_revno(self):
        if False:
            print('Hello World!')
        tree = self.create_tree_with_merge()
        the_branch = tree.branch
        self.assertEqual((0,), the_branch.revision_id_to_dotted_revno('null:'))
        self.assertEqual((1,), the_branch.revision_id_to_dotted_revno('rev-1'))
        self.assertEqual((2,), the_branch.revision_id_to_dotted_revno('rev-2'))
        self.assertEqual((3,), the_branch.revision_id_to_dotted_revno('rev-3'))
        self.assertEqual((1, 1, 1), the_branch.revision_id_to_dotted_revno('rev-1.1.1'))
        self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_dotted_revno, 'rev-1.0.2')