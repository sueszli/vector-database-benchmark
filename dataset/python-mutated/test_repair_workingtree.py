from bzrlib import workingtree
from bzrlib.tests import TestCaseWithTransport

class TestRepairWorkingTree(TestCaseWithTransport):

    def break_dirstate(self, tree, completely=False):
        if False:
            return 10
        'Write garbage into the dirstate file.'
        self.assertIsNot(None, getattr(tree, 'current_dirstate', None))
        tree.lock_read()
        try:
            dirstate = tree.current_dirstate()
            dirstate_path = dirstate._filename
            self.assertPathExists(dirstate_path)
        finally:
            tree.unlock()
        if completely:
            f = open(dirstate_path, 'wb')
        else:
            f = open(dirstate_path, 'ab')
        try:
            f.write('garbage-at-end-of-file\n')
        finally:
            f.close()

    def make_initial_tree(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo', 'tree/dir/', 'tree/dir/bar'])
        tree.add(['foo', 'dir', 'dir/bar'])
        tree.commit('first')
        return tree

    def test_repair_refuses_uncorrupted(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_initial_tree()
        self.run_bzr_error(['The tree does not appear to be corrupt', '"bzr revert"', '--force'], 'repair-workingtree -d tree')

    def test_repair_forced(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_initial_tree()
        tree.rename_one('dir', 'alt_dir')
        self.assertIsNot(None, tree.path2id('alt_dir'))
        self.run_bzr('repair-workingtree -d tree --force')
        self.assertIs(None, tree.path2id('alt_dir'))
        self.assertPathExists('tree/alt_dir')

    def test_repair_corrupted_dirstate(self):
        if False:
            print('Hello World!')
        tree = self.make_initial_tree()
        self.break_dirstate(tree)
        self.run_bzr('repair-workingtree -d tree')
        tree = workingtree.WorkingTree.open('tree')
        tree.check_state()

    def test_repair_naive_destroyed_fails(self):
        if False:
            return 10
        tree = self.make_initial_tree()
        self.break_dirstate(tree, completely=True)
        self.run_bzr_error(['the header appears corrupt, try passing'], 'repair-workingtree -d tree')

    def test_repair_destroyed_with_revs_passes(self):
        if False:
            while True:
                i = 10
        tree = self.make_initial_tree()
        self.break_dirstate(tree, completely=True)
        self.run_bzr('repair-workingtree -d tree -r -1')
        tree = workingtree.WorkingTree.open('tree')
        tree.check_state()