"""Tests for interface conformance of 'WorkingTree.rename_one'"""
import os
from bzrlib import errors, osutils, tests
from bzrlib.tests import features
from bzrlib.tests.matchers import HasLayout
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestRenameOne(TestCaseWithWorkingTree):

    def assertTreeLayout(self, expected, tree):
        if False:
            i = 10
            return i + 15
        'Check that the tree has the correct layout.'
        self.assertThat(tree, HasLayout(expected))

    def test_rename_one_target_not_dir(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'not-a-dir/b')

    def test_rename_one_non_existent(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/'])
        tree.add(['a'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'not-a-file', 'a/failure')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'not-a-file', 'also_not')

    def test_rename_one_target_not_versioned(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'b'])
        tree.add(['b'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'b', 'a/b')

    def test_rename_one_unversioned(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'b'])
        tree.add(['a'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'b', 'a/b')

    def test_rename_one_samedir(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        a_contents = tree.get_file_text('a-id')
        tree.rename_one('a', 'foo')
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())
        self.assertPathDoesNotExist('a')
        self.assertFileEqual(a_contents, 'foo')

    def test_rename_one_not_localdir(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a', 'tree/b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        a_contents = tree.get_file_text('a-id')
        tree.rename_one('a', 'b/foo')
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())
        self.assertPathDoesNotExist('tree/a')
        self.assertFileEqual(a_contents, 'tree/b/foo')

    def test_rename_one_subdir(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/c'])
        tree.add(['a', 'b', 'b/c'], ['a-id', 'b-id', 'c-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('b/c', 'c-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('b/c', 'c-id')], tree.basis_tree())
        a_contents = tree.get_file_text('a-id')
        tree.rename_one('a', 'b/d')
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/c', 'c-id'), ('b/d', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('b/c', 'c-id')], tree.basis_tree())
        self.assertPathDoesNotExist('a')
        self.assertFileEqual(a_contents, 'b/d')

    def test_rename_one_parent_dir(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/c'])
        tree.add(['a', 'b', 'b/c'], ['a-id', 'b-id', 'c-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        c_contents = tree.get_file_text('c-id')
        tree.rename_one('b/c', 'd')
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('d', 'c-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('b/c', 'c-id')], tree.basis_tree())
        self.assertPathDoesNotExist('b/c')
        self.assertFileEqual(c_contents, 'd')

    def test_rename_one_fail_consistent(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/a', 'c'])
        tree.add(['a', 'b', 'c'], ['a-id', 'b-id', 'c-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        self.assertRaises(errors.RenameFailedFilesExist, tree.rename_one, 'a', 'b/a')
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('c', 'c-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id'), ('c', 'c-id')], tree.basis_tree())

    def test_rename_one_onto_existing(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'b')

    def test_rename_one_onto_self(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['b/', 'b/a'])
        tree.add(['b', 'b/a'], ['b-id', 'a-id'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'b/a', 'b/a')

    def test_rename_one_onto_self_root(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'], ['a-id'])
        tree.commit('initial', rev_id='rev-1')
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'a')

    def test_rename_one_after(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.rename('a', 'b/foo')
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree)
        tree.rename_one('a', 'b/foo')
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())

    def test_rename_one_after_with_after(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.rename('a', 'b/foo')
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree)
        tree.rename_one('a', 'b/foo', after=True)
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())

    def test_rename_one_after_dest_versioned(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'], ['a-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.rename('a', 'b')
        tree.add(['b'], ['b-id'])
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b', 'b-id')], tree)
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'b')
        self.assertIsInstance(e.extra, errors.AlreadyVersionedError)

    def test_rename_one_after_with_after_dest_versioned(self):
        if False:
            i = 10
            return i + 15
        ' using after with an already versioned file should fail '
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.unlink('a')
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b', 'b-id')], tree)
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'b', after=True)
        self.assertIsInstance(e.extra, errors.AlreadyVersionedError)

    def test_rename_one_after_with_after_dest_added(self):
        if False:
            print('Hello World!')
        ' using after with a newly added file should work '
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'], ['a-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.rename('a', 'b')
        tree.add(['b'], ['b-id'])
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b', 'b-id')], tree)
        tree.rename_one('a', 'b', after=True)
        self.assertTreeLayout([('', root_id), ('b', 'a-id')], tree)

    def test_rename_one_after_source_removed(self):
        if False:
            return 10
        'Rename even if the source was already unversioned.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        os.rename('a', 'b/foo')
        tree.remove(['a'])
        self.assertTreeLayout([('', root_id), ('b/', 'b-id')], tree)
        tree.rename_one('a', 'b/foo')
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())

    def test_rename_one_after_no_target(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'b/foo', after=True)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())

    def test_rename_one_after_source_and_dest(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/foo'])
        tree.add(['a', 'b'], ['a-id', 'b-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        a_file = open('a', 'r')
        try:
            a_text = a_file.read()
        finally:
            a_file.close()
        foo_file = open('b/foo', 'r')
        try:
            foo_text = foo_file.read()
        finally:
            foo_file.close()
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree)
        self.assertRaises(errors.RenameFailedFilesExist, tree.rename_one, 'a', 'b/foo', after=False)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree)
        self.assertFileEqual(a_text, 'a')
        self.assertFileEqual(foo_text, 'b/foo')
        tree.rename_one('a', 'b/foo', after=True)
        self.assertTreeLayout([('', root_id), ('b/', 'b-id'), ('b/foo', 'a-id')], tree)
        self.assertTreeLayout([('', root_id), ('a', 'a-id'), ('b/', 'b-id')], tree.basis_tree())
        self.assertFileEqual(a_text, 'a')
        self.assertFileEqual(foo_text, 'b/foo')

    def test_rename_one_directory(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'a/c/', 'a/c/d', 'e/'])
        tree.add(['a', 'a/b', 'a/c', 'a/c/d', 'e'], ['a-id', 'b-id', 'c-id', 'd-id', 'e-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        tree.rename_one('a', 'e/f')
        self.assertTreeLayout([('', root_id), ('e/', 'e-id'), ('e/f/', 'a-id'), ('e/f/b', 'b-id'), ('e/f/c/', 'c-id'), ('e/f/c/d', 'd-id')], tree)
        self.assertTreeLayout([('', root_id), ('a/', 'a-id'), ('e/', 'e-id'), ('a/b', 'b-id'), ('a/c/', 'c-id'), ('a/c/d', 'd-id')], tree.basis_tree())

    def test_rename_one_moved(self):
        if False:
            return 10
        'Moving a moved entry works as expected.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'c/'])
        tree.add(['a', 'a/b', 'c'], ['a-id', 'b-id', 'c-id'])
        tree.commit('initial', rev_id='rev-1')
        root_id = tree.get_root_id()
        tree.rename_one('a/b', 'c/foo')
        self.assertTreeLayout([('', root_id), ('a/', 'a-id'), ('c/', 'c-id'), ('c/foo', 'b-id')], tree)
        self.assertTreeLayout([('', root_id), ('a/', 'a-id'), ('c/', 'c-id'), ('a/b', 'b-id')], tree.basis_tree())
        tree.rename_one('c/foo', 'bar')
        self.assertTreeLayout([('', root_id), ('a/', 'a-id'), ('bar', 'b-id'), ('c/', 'c-id')], tree)
        self.assertTreeLayout([('', root_id), ('a/', 'a-id'), ('c/', 'c-id'), ('a/b', 'b-id')], tree.basis_tree())

    def test_rename_to_denormalised_fails(self):
        if False:
            while True:
                i = 10
        if osutils.normalizes_filenames():
            raise tests.TestNotApplicable('OSX normalizes filenames')
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'])
        self.assertRaises((errors.InvalidNormalization, UnicodeEncodeError), tree.rename_one, 'a', u'bårry')

    def test_rename_unversioned_non_ascii(self):
        if False:
            return 10
        'Check error when renaming an unversioned non-ascii file'
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree([u'§'])
        e = self.assertRaises(errors.BzrRenameFailedError, tree.rename_one, u'§', 'b')
        self.assertIsInstance(e.extra, errors.NotVersionedError)
        self.assertEqual(e.extra.path, u'§')

    def test_rename_into_unversioned_non_ascii_dir(self):
        if False:
            print('Hello World!')
        'Check error when renaming into unversioned non-ascii directory'
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', u'§/'])
        tree.add(['a'])
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', u'§/a')
        self.assertIsInstance(e.extra, errors.NotVersionedError)
        self.assertEqual(e.extra.path, u'§')

    def test_rename_over_already_versioned_non_ascii(self):
        if False:
            i = 10
            return i + 15
        'Check error renaming over an already versioned non-ascii file'
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', u'§'])
        tree.add(['a', u'§'])
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', u'§')
        self.assertIsInstance(e.extra, errors.AlreadyVersionedError)
        self.assertEqual(e.extra.path, u'§')

    def test_rename_after_non_existant_non_ascii(self):
        if False:
            print('Hello World!')
        'Check error renaming after move with missing non-ascii file'
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'])
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', u'§', after=True)
        self.assertIsInstance(e.extra, errors.NoSuchFile)
        self.assertEqual(e.extra.path, u'§')