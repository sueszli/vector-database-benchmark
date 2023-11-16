"""Tests of the WorkingTree.unversion API."""
from bzrlib import errors
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestUnversion(TestCaseWithWorkingTree):

    def test_unversion_requires_write_lock(self):
        if False:
            return 10
        'WT.unversion([]) in a read lock raises ReadOnlyError.'
        tree = self.make_branch_and_tree('.')
        tree.lock_read()
        self.assertRaises(errors.ReadOnlyError, tree.unversion, [])
        tree.unlock()

    def test_unversion_missing_file(self):
        if False:
            return 10
        "WT.unversion(['missing-id']) raises NoSuchId."
        tree = self.make_branch_and_tree('.')
        self.assertRaises(errors.NoSuchId, tree.unversion, ['missing-id'])

    def test_unversion_parent_and_child_renamed_bug_187207(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['del/', 'del/sub/', 'del/sub/b'])
        tree.add(['del', 'del/sub', 'del/sub/b'], ['del', 'sub', 'b'])
        tree.commit('setup')
        tree.rename_one('del/sub', 'sub')
        self.assertEqual('sub/b', tree.id2path('b'))
        tree.unversion(['del', 'b'])
        self.assertRaises(errors.NoSuchId, tree.id2path, 'b')

    def test_unversion_several_files(self):
        if False:
            print('Hello World!')
        'After unversioning several files, they should not be versioned.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b', 'c'])
        tree.add(['a', 'b', 'c'], ['a-id', 'b-id', 'c-id'])
        tree.lock_write()
        tree.unversion(['a-id', 'b-id'])
        self.assertFalse(tree.has_id('a-id'))
        self.assertFalse(tree.has_id('b-id'))
        self.assertTrue(tree.has_id('c-id'))
        self.assertTrue(tree.has_filename('a'))
        self.assertTrue(tree.has_filename('b'))
        self.assertTrue(tree.has_filename('c'))
        tree.unlock()
        tree = tree.bzrdir.open_workingtree()
        tree.lock_read()
        self.assertFalse(tree.has_id('a-id'))
        self.assertFalse(tree.has_id('b-id'))
        self.assertTrue(tree.has_id('c-id'))
        self.assertTrue(tree.has_filename('a'))
        self.assertTrue(tree.has_filename('b'))
        self.assertTrue(tree.has_filename('c'))
        tree.unlock()

    def test_unversion_subtree(self):
        if False:
            while True:
                i = 10
        'Unversioning the root of a subtree unversions the entire subtree.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'c'])
        tree.add(['a', 'a/b', 'c'], ['a-id', 'b-id', 'c-id'])
        tree.lock_write()
        tree.unversion(['a-id'])
        self.assertFalse(tree.has_id('a-id'))
        self.assertFalse(tree.has_id('b-id'))
        self.assertTrue(tree.has_id('c-id'))
        self.assertTrue(tree.has_filename('a'))
        self.assertTrue(tree.has_filename('a/b'))
        self.assertTrue(tree.has_filename('c'))
        tree.unlock()

    def test_unversion_subtree_and_children(self):
        if False:
            i = 10
            return i + 15
        'Passing a child id will raise NoSuchId.\n\n        This is because the parent directory will have already been removed.\n        '
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'a/c', 'd'])
        tree.add(['a', 'a/b', 'a/c', 'd'], ['a-id', 'b-id', 'c-id', 'd-id'])
        tree.lock_write()
        try:
            tree.unversion(['b-id', 'a-id'])
            self.assertFalse(tree.has_id('a-id'))
            self.assertFalse(tree.has_id('b-id'))
            self.assertFalse(tree.has_id('c-id'))
            self.assertTrue(tree.has_id('d-id'))
            self.assertTrue(tree.has_filename('a'))
            self.assertTrue(tree.has_filename('a/b'))
            self.assertTrue(tree.has_filename('a/c'))
            self.assertTrue(tree.has_filename('d'))
        finally:
            tree.unlock()

    def test_unversion_renamed(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/dir/', 'a/dir/f1', 'a/dir/f2', 'a/dir/f3', 'a/dir2/'])
        tree.add(['dir', 'dir/f1', 'dir/f2', 'dir/f3', 'dir2'], ['dir-id', 'f1-id', 'f2-id', 'f3-id', 'dir2-id'])
        rev_id1 = tree.commit('init')
        tree.rename_one('dir/f1', 'dir/a')
        tree.rename_one('dir/f2', 'dir/z')
        tree.move(['dir/f3'], 'dir2')
        tree.lock_read()
        try:
            root_id = tree.get_root_id()
            paths = [(path, ie.file_id) for (path, ie) in tree.iter_entries_by_dir()]
        finally:
            tree.unlock()
        self.assertEqual([('', root_id), ('dir', 'dir-id'), ('dir2', 'dir2-id'), ('dir/a', 'f1-id'), ('dir/z', 'f2-id'), ('dir2/f3', 'f3-id')], paths)
        tree.unversion(set(['dir-id']))
        paths = [(path, ie.file_id) for (path, ie) in tree.iter_entries_by_dir()]
        self.assertEqual([('', root_id), ('dir2', 'dir2-id'), ('dir2/f3', 'f3-id')], paths)

    def test_unversion_after_conflicted_merge(self):
        if False:
            return 10
        tree_a = self.make_branch_and_tree('A')
        self.build_tree(['A/a/', 'A/a/m', 'A/a/n'])
        tree_a.add(['a', 'a/m', 'a/n'], ['a-id', 'm-id', 'n-id'])
        tree_a.commit('init')
        tree_a.lock_read()
        try:
            root_id = tree_a.get_root_id()
        finally:
            tree_a.unlock()
        tree_b = tree_a.bzrdir.sprout('B').open_workingtree()
        self.build_tree(['B/xyz/'])
        tree_b.add(['xyz'], ['xyz-id'])
        tree_b.rename_one('a/m', 'xyz/m')
        tree_b.unversion(['a-id'])
        tree_b.commit('delete in B')
        paths = [(path, ie.file_id) for (path, ie) in tree_b.iter_entries_by_dir()]
        self.assertEqual([('', root_id), ('xyz', 'xyz-id'), ('xyz/m', 'm-id')], paths)
        self.build_tree_contents([('A/a/n', 'new contents for n\n')])
        tree_a.commit('change n in A')
        num_conflicts = tree_b.merge_from_branch(tree_a.branch)
        self.assertEqual(4, num_conflicts)
        paths = [(path, ie.file_id) for (path, ie) in tree_b.iter_entries_by_dir()]
        self.assertEqual([('', root_id), ('a', 'a-id'), ('xyz', 'xyz-id'), ('a/n.OTHER', 'n-id'), ('xyz/m', 'm-id')], paths)
        tree_b.unversion(['a-id'])
        paths = [(path, ie.file_id) for (path, ie) in tree_b.iter_entries_by_dir()]
        self.assertEqual([('', root_id), ('xyz', 'xyz-id'), ('xyz/m', 'm-id')], paths)