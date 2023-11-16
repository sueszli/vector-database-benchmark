"""Tests for WorkingTree.paths2ids.

This API probably needs to be exposed as a tree implementation test, but these
initial tests are for the specific cases being refactored from
find_ids_across_trees.
"""
from operator import attrgetter
from bzrlib import errors
from bzrlib.tests import features
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestPaths2Ids(TestCaseWithWorkingTree):

    def assertExpectedIds(self, ids, tree, paths, trees=None, require_versioned=True):
        if False:
            print('Hello World!')
        'Run paths2ids for tree, and check the result.'
        tree.lock_read()
        if trees:
            map(apply, map(attrgetter('lock_read'), trees))
            result = tree.paths2ids(paths, trees, require_versioned=require_versioned)
            map(apply, map(attrgetter('unlock'), trees))
        else:
            result = tree.paths2ids(paths, require_versioned=require_versioned)
        self.assertEqual(set(ids), result)
        tree.unlock()

    def test_paths_none_result_none(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        tree.lock_read()
        self.assertEqual(None, tree.paths2ids(None))
        tree.unlock()

    def test_find_single_root(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.assertExpectedIds([tree.path2id('')], tree, [''])

    def test_find_tree_and_clone_roots(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        clone = tree.bzrdir.clone('clone').open_workingtree()
        clone.lock_tree_write()
        clone_root_id = 'new-id'
        clone.set_root_id(clone_root_id)
        tree_root_id = tree.path2id('')
        clone.unlock()
        self.assertExpectedIds([tree_root_id, clone_root_id], tree, [''], [clone])

    def test_find_tree_basis_roots(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        tree.commit('basis')
        basis = tree.basis_tree()
        basis_root_id = basis.path2id('')
        tree.lock_tree_write()
        tree_root_id = 'new-id'
        tree.set_root_id(tree_root_id)
        tree.unlock()
        self.assertExpectedIds([tree_root_id, basis_root_id], tree, [''], [basis])

    def test_find_children_of_moved_directories(self):
        if False:
            print('Hello World!')
        "Check the basic nasty corner case that path2ids should handle.\n\n        This is the following situation:\n        basis:\n          / ROOT\n          /dir dir\n          /dir/child-moves child-moves\n          /dir/child-stays child-stays\n          /dir/child-goes  child-goes\n\n        current tree:\n          / ROOT\n          /child-moves child-moves\n          /newdir newdir\n          /newdir/dir  dir\n          /newdir/dir/child-stays child-stays\n          /newdir/dir/new-child   new-child\n\n        In english: we move a directory under a directory that was a sibling,\n        and at the same time remove, or move out of the directory, some of its\n        children, and give it a new child previous absent or a sibling.\n\n        current_tree.path2ids(['newdir'], [basis]) is meant to handle this\n        correctly: that is it should return the ids:\n          newdir because it was provided\n          dir, because its under newdir in current\n          child-moves because its under dir in old\n          child-stays either because its under newdir/dir in current, or under dir in old\n          child-goes because its under dir in old.\n          new-child because its under dir in new\n\n        Symmetrically, current_tree.path2ids(['dir'], [basis]) is meant to show\n        new-child, even though its not under the path 'dir' in current, because\n        its under a path selected by 'dir' in basis:\n          dir because its selected in basis.\n          child-moves because its under dir in old\n          child-stays either because its under newdir/dir in current, or under dir in old\n          child-goes because its under dir in old.\n          new-child because its under dir in new.\n        "
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/dir/', 'tree/dir/child-moves', 'tree/dir/child-stays', 'tree/dir/child-goes'])
        tree.add(['dir', 'dir/child-moves', 'dir/child-stays', 'dir/child-goes'], ['dir', 'child-moves', 'child-stays', 'child-goes'])
        tree.commit('create basis')
        basis = tree.basis_tree()
        tree.unversion(['child-goes'])
        tree.rename_one('dir/child-moves', 'child-moves')
        self.build_tree(['tree/newdir/'])
        tree.add(['newdir'], ['newdir'])
        tree.rename_one('dir/child-stays', 'child-stays')
        tree.rename_one('dir', 'newdir/dir')
        tree.rename_one('child-stays', 'newdir/dir/child-stays')
        self.build_tree(['tree/newdir/dir/new-child'])
        tree.add(['newdir/dir/new-child'], ['new-child'])
        self.assertExpectedIds(['newdir', 'dir', 'child-moves', 'child-stays', 'child-goes', 'new-child'], tree, ['newdir'], [basis])
        self.assertExpectedIds(['dir', 'child-moves', 'child-stays', 'child-goes', 'new-child'], tree, ['dir'], [basis])

    def test_unversioned_one_tree(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/unversioned'])
        self.assertExpectedIds([], tree, ['unversioned'], require_versioned=False)
        tree.lock_read()
        self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, ['unversioned'])
        tree.unlock()

    def test_unversioned_in_one_of_multiple_trees(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        tree.commit('make basis')
        basis = tree.basis_tree()
        self.build_tree(['tree/in-one'])
        tree.add(['in-one'], ['in-one'])
        self.assertExpectedIds(['in-one'], tree, ['in-one'], [basis])

    def test_unversioned_all_of_multiple_trees(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        tree.commit('make basis')
        basis = tree.basis_tree()
        self.assertExpectedIds([], tree, ['unversioned'], [basis], require_versioned=False)
        tree.lock_read()
        basis.lock_read()
        self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, ['unversioned'], [basis])
        self.assertRaises(errors.PathsNotVersionedError, basis.paths2ids, ['unversioned'], [tree])
        basis.unlock()
        tree.unlock()

    def test_unversioned_non_ascii_one_tree(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree([u'§'])
        self.assertExpectedIds([], tree, [u'§'], require_versioned=False)
        self.addCleanup(tree.lock_read().unlock)
        e = self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, [u'§'])
        self.assertEqual([u'§'], e.paths)