"""Test that WorkingTree.basis_tree() yields a valid tree."""
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestBasisTree(TestCaseWithWorkingTree):

    def test_emtpy_tree(self):
        if False:
            print('Hello World!')
        'A working tree with no parents.'
        tree = self.make_branch_and_tree('tree')
        basis_tree = tree.basis_tree()
        basis_tree.lock_read()
        try:
            self.assertEqual([], list(basis_tree.list_files(include_root=True)))
        finally:
            basis_tree.unlock()

    def test_same_tree(self):
        if False:
            i = 10
            return i + 15
        "Test basis_tree when working tree hasn't been modified."
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file', 'dir/', 'dir/subfile'])
        tree.add(['file', 'dir', 'dir/subfile'])
        revision_id = tree.commit('initial tree')
        basis_tree = tree.basis_tree()
        basis_tree.lock_read()
        try:
            self.assertEqual(revision_id, basis_tree.get_revision_id())
            self.assertEqual(['', 'dir', 'dir/subfile', 'file'], sorted((info[0] for info in basis_tree.list_files(True))))
        finally:
            basis_tree.unlock()

    def test_altered_tree(self):
        if False:
            return 10
        'Test basis really is basis after working has been modified.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file', 'dir/', 'dir/subfile'])
        tree.add(['file', 'dir', 'dir/subfile'])
        revision_id = tree.commit('initial tree')
        self.build_tree(['new file', 'new dir/'])
        tree.rename_one('file', 'dir/new file')
        tree.unversion([tree.path2id('dir/subfile')])
        tree.add(['new file', 'new dir'])
        basis_tree = tree.basis_tree()
        basis_tree.lock_read()
        try:
            self.assertEqual(revision_id, basis_tree.get_revision_id())
            self.assertEqual(['', 'dir', 'dir/subfile', 'file'], sorted((info[0] for info in basis_tree.list_files(True))))
        finally:
            basis_tree.unlock()