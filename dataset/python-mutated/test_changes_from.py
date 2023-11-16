"""Test Tree.changes_from() for WorkingTree specific scenarios."""
from bzrlib import revision
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestChangesFrom(TestCaseWithWorkingTree):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestChangesFrom, self).setUp()
        self.tree = self.make_branch_and_tree('tree')
        files = ['a', 'b/', 'b/c']
        self.build_tree(files, transport=self.tree.bzrdir.root_transport)
        self.tree.add(files, ['a-id', 'b-id', 'c-id'])
        self.tree.commit('initial tree')

    def test_unknown(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['tree/unknown'])
        d = self.tree.changes_from(self.tree.basis_tree())
        self.assertEqual([], d.added)
        self.assertEqual([], d.removed)
        self.assertEqual([], d.renamed)
        self.assertEqual([], d.modified)

    def test_unknown_specific_file(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(['tree/unknown'])
        empty_tree = self.tree.branch.repository.revision_tree(revision.NULL_REVISION)
        d = self.tree.changes_from(empty_tree, specific_files=['unknown'])
        self.assertEqual([], d.added)
        self.assertEqual([], d.removed)
        self.assertEqual([], d.renamed)
        self.assertEqual([], d.modified)