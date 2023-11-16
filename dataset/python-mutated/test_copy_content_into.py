"""Tests for bzrlib.branch.InterBranch.copy_content_into."""
from bzrlib import branch
from bzrlib.tests.per_interbranch import StubMatchingInter, StubWithFormat, TestCaseWithInterBranch

class TestCopyContentInto(TestCaseWithInterBranch):

    def test_contract_convenience_method(self):
        if False:
            return 10
        self.tree1 = self.make_from_branch_and_tree('tree1')
        rev1 = self.tree1.commit('one')
        branch2 = self.make_to_branch('tree2')
        branch2.repository.fetch(self.tree1.branch.repository)
        self.tree1.branch.copy_content_into(branch2, revision_id=rev1)

    def test_inter_is_used(self):
        if False:
            print('Hello World!')
        self.tree1 = self.make_from_branch_and_tree('tree1')
        self.addCleanup(branch.InterBranch.unregister_optimiser, StubMatchingInter)
        branch.InterBranch.register_optimiser(StubMatchingInter)
        del StubMatchingInter._uses[:]
        self.tree1.branch.copy_content_into(StubWithFormat(), revision_id='54')
        self.assertLength(1, StubMatchingInter._uses)
        use = StubMatchingInter._uses[0]
        self.assertEqual('copy_content_into', use[1])
        self.assertEqual('54', use[3]['revision_id'])
        del StubMatchingInter._uses[:]