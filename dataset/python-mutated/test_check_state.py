"""Tests for WorkingTree.check_state."""
from bzrlib import errors, tests
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestCaseWithState(TestCaseWithWorkingTree):

    def make_tree_with_broken_dirstate(self, path):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree(path)
        self.break_dirstate(tree)
        return tree

    def break_dirstate(self, tree, completely=False):
        if False:
            i = 10
            return i + 15
        'Write garbage into the dirstate file.'
        if getattr(tree, 'current_dirstate', None) is None:
            raise tests.TestNotApplicable('Only applies to dirstate-based trees')
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

class TestCheckState(TestCaseWithState):

    def test_check_state(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        tree.check_state()

    def test_check_broken_dirstate(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_tree_with_broken_dirstate('tree')
        self.assertRaises(errors.BzrError, tree.check_state)

class TestResetState(TestCaseWithState):

    def make_initial_tree(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo', 'tree/dir/', 'tree/dir/bar'])
        tree.add(['foo', 'dir', 'dir/bar'])
        tree.commit('initial')
        return tree

    def test_reset_state_forgets_changes(self):
        if False:
            while True:
                i = 10
        tree = self.make_initial_tree()
        foo_id = tree.path2id('foo')
        tree.rename_one('foo', 'baz')
        self.assertEqual(None, tree.path2id('foo'))
        self.assertEqual(foo_id, tree.path2id('baz'))
        tree.reset_state()
        self.assertEqual(foo_id, tree.path2id('foo'))
        self.assertEqual(None, tree.path2id('baz'))
        self.assertPathDoesNotExist('tree/foo')
        self.assertPathExists('tree/baz')

    def test_reset_state_handles_corrupted_dirstate(self):
        if False:
            while True:
                i = 10
        tree = self.make_initial_tree()
        rev_id = tree.last_revision()
        self.break_dirstate(tree)
        tree.reset_state()
        tree.check_state()
        self.assertEqual(rev_id, tree.last_revision())

    def test_reset_state_handles_destroyed_dirstate(self):
        if False:
            while True:
                i = 10
        tree = self.make_initial_tree()
        rev_id = tree.last_revision()
        self.break_dirstate(tree, completely=True)
        tree.reset_state(revision_ids=[rev_id])
        tree.check_state()
        self.assertEqual(rev_id, tree.last_revision())