"""Test symlink support.
"""
import os
from bzrlib import osutils, tests, workingtree
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree
from bzrlib.tests import features

class TestSmartAddTree(TestCaseWithWorkingTree):
    _test_needs_features = [features.SymlinkFeature]

    def test_smart_add_symlink(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', 'target')])
        tree.smart_add(['tree/link'])
        self.assertIsNot(None, tree.path2id('link'))
        self.assertIs(None, tree.path2id('target'))
        self.assertEqual('symlink', tree.kind(tree.path2id('link')))

    def test_smart_add_symlink_pointing_outside(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', '../../../../target')])
        tree.smart_add(['tree/link'])
        self.assertIsNot(None, tree.path2id('link'))
        self.assertIs(None, tree.path2id('target'))
        self.assertEqual('symlink', tree.kind(tree.path2id('link')))

    def test_add_file_under_symlink(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', 'dir'), ('tree/dir/',), ('tree/dir/file', 'content')])
        self.assertEqual(tree.smart_add(['tree/link/file']), ([u'dir', u'dir/file'], {}))
        self.assertTrue(tree.path2id('dir/file'))
        self.assertTrue(tree.path2id('dir'))
        self.assertIs(None, tree.path2id('link'))
        self.assertIs(None, tree.path2id('link/file'))

class TestKindChanges(TestCaseWithWorkingTree):
    _test_needs_features = [features.SymlinkFeature]

    def test_symlink_changes_to_dir(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a@', 'target')])
        tree.smart_add(['tree/a'])
        tree.commit('add symlink')
        os.unlink('tree/a')
        self.build_tree_contents([('tree/a/',), ('tree/a/f', 'content')])
        tree.smart_add(['tree/a/f'])
        tree.commit('change to dir')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual([], list(tree.iter_changes(tree.basis_tree())))
        if tree._format.supports_versioned_directories:
            self.assertEqual(['a', 'a/f'], sorted((info[0] for info in tree.list_files())))
        else:
            self.assertEqual([], list(tree.list_files()))

    def test_dir_changes_to_symlink(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a/',), ('tree/a/file', 'content')])
        tree.smart_add(['tree/a'])
        tree.commit('add dir')
        osutils.rmtree('tree/a')
        self.build_tree_contents([('tree/a@', 'target')])
        tree.commit('change to symlink')

class TestOpenTree(TestCaseWithWorkingTree):
    _test_needs_features = [features.SymlinkFeature]

    def test_open_containing_through_symlink(self):
        if False:
            print('Hello World!')
        self.make_test_tree()
        self.check_open_containing('link/content', 'tree', 'content')
        self.check_open_containing('link/sublink', 'tree', 'sublink')
        self.check_open_containing('link/sublink/subcontent', 'tree', 'sublink/subcontent')

    def check_open_containing(self, to_open, expected_tree_name, expected_relpath):
        if False:
            print('Hello World!')
        (wt, relpath) = workingtree.WorkingTree.open_containing(to_open)
        self.assertEqual(relpath, expected_relpath)
        self.assertEndsWith(wt.basedir, expected_tree_name)

    def test_tree_files(self):
        if False:
            return 10
        self.make_test_tree()
        self.check_tree_files(['tree/outerlink'], 'tree', ['outerlink'])
        self.check_tree_files(['link/outerlink'], 'tree', ['outerlink'])
        self.check_tree_files(['link/sublink/subcontent'], 'tree', ['subdir/subcontent'])

    def check_tree_files(self, to_open, expected_tree, expect_paths):
        if False:
            i = 10
            return i + 15
        (tree, relpaths) = workingtree.WorkingTree.open_containing_paths(to_open)
        self.assertEndsWith(tree.basedir, expected_tree)
        self.assertEqual(expect_paths, relpaths)

    def make_test_tree(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('link@', 'tree'), ('tree/outerlink@', '/not/there'), ('tree/content', 'hello'), ('tree/sublink@', 'subdir'), ('tree/subdir/',), ('tree/subdir/subcontent', 'subcontent stuff')])