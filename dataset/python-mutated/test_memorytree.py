"""Tests for the MemoryTree class."""
from bzrlib import errors
from bzrlib.memorytree import MemoryTree
from bzrlib.tests import TestCaseWithTransport
from bzrlib.treebuilder import TreeBuilder

class TestMemoryTree(TestCaseWithTransport):

    def test_create_on_branch(self):
        if False:
            while True:
                i = 10
        'Creating a mutable tree on a trivial branch works.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        self.assertEqual(branch.bzrdir, tree.bzrdir)
        self.assertEqual(branch, tree.branch)
        self.assertEqual([], tree.get_parent_ids())

    def test_create_on_branch_with_content(self):
        if False:
            return 10
        'Creating a mutable tree on a non-trivial branch works.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        builder = TreeBuilder()
        builder.start_tree(tree)
        builder.build(['foo'])
        builder.finish_tree()
        rev_id = tree.commit('first post')
        tree.unlock()
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_read()
        self.assertEqual([rev_id], tree.get_parent_ids())
        self.assertEqual('contents of foo\n', tree.get_file(tree.path2id('foo')).read())
        tree.unlock()

    def test_get_root_id(self):
        if False:
            i = 10
            return i + 15
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        try:
            tree.add([''])
            self.assertIsNot(None, tree.get_root_id())
        finally:
            tree.unlock()

    def test_lock_tree_write(self):
        if False:
            print('Hello World!')
        'Check we can lock_tree_write and unlock MemoryTrees.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_tree_write()
        tree.unlock()

    def test_lock_tree_write_after_read_fails(self):
        if False:
            return 10
        'Check that we error when trying to upgrade a read lock to write.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_read()
        self.assertRaises(errors.ReadOnlyError, tree.lock_tree_write)
        tree.unlock()

    def test_lock_write(self):
        if False:
            return 10
        'Check we can lock_write and unlock MemoryTrees.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.unlock()

    def test_lock_write_after_read_fails(self):
        if False:
            i = 10
            return i + 15
        'Check that we error when trying to upgrade a read lock to write.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_read()
        self.assertRaises(errors.ReadOnlyError, tree.lock_write)
        tree.unlock()

    def test_add_with_kind(self):
        if False:
            print('Hello World!')
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.add(['', 'afile', 'adir'], None, ['directory', 'file', 'directory'])
        self.assertEqual('afile', tree.id2path(tree.path2id('afile')))
        self.assertEqual('adir', tree.id2path(tree.path2id('adir')))
        self.assertFalse(tree.has_filename('afile'))
        self.assertFalse(tree.has_filename('adir'))
        tree.unlock()

    def test_put_new_file(self):
        if False:
            for i in range(10):
                print('nop')
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.add(['', 'foo'], ids=['root-id', 'foo-id'], kinds=['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo-id', 'barshoom')
        self.assertEqual('barshoom', tree.get_file('foo-id').read())
        tree.unlock()

    def test_put_existing_file(self):
        if False:
            return 10
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.add(['', 'foo'], ids=['root-id', 'foo-id'], kinds=['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo-id', 'first-content')
        tree.put_file_bytes_non_atomic('foo-id', 'barshoom')
        self.assertEqual('barshoom', tree.get_file('foo-id').read())
        tree.unlock()

    def test_add_in_subdir(self):
        if False:
            while True:
                i = 10
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ['root-id'], ['directory'])
        tree.mkdir('adir', 'dir-id')
        tree.add(['adir/afile'], ['file-id'], ['file'])
        self.assertEqual('adir/afile', tree.id2path('file-id'))
        self.assertEqual('adir', tree.id2path('dir-id'))
        tree.put_file_bytes_non_atomic('file-id', 'barshoom')

    def test_commit_trivial(self):
        if False:
            i = 10
            return i + 15
        'Smoke test for commit on a MemoryTree.\n\n        Becamse of commits design and layering, if this works, all commit\n        logic should work quite reliably.\n        '
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.add(['', 'foo'], ids=['root-id', 'foo-id'], kinds=['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo-id', 'barshoom')
        revision_id = tree.commit('message baby')
        self.assertEqual([revision_id], tree.get_parent_ids())
        tree.unlock()
        revtree = tree.branch.repository.revision_tree(revision_id)
        revtree.lock_read()
        self.addCleanup(revtree.unlock)
        self.assertEqual('barshoom', revtree.get_file('foo-id').read())

    def test_unversion(self):
        if False:
            print('Hello World!')
        'Some test for unversion of a memory tree.'
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        tree.lock_write()
        tree.add(['', 'foo'], ids=['root-id', 'foo-id'], kinds=['directory', 'file'])
        tree.unversion(['foo-id'])
        self.assertFalse(tree.has_id('foo-id'))
        tree.unlock()

    def test_last_revision(self):
        if False:
            while True:
                i = 10
        'There should be a last revision method we can call.'
        tree = self.make_branch_and_memory_tree('branch')
        tree.lock_write()
        tree.add('')
        rev_id = tree.commit('first post')
        tree.unlock()
        self.assertEqual(rev_id, tree.last_revision())

    def test_rename_file(self):
        if False:
            return 10
        tree = self.make_branch_and_memory_tree('branch')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add(['', 'foo'], ['root-id', 'foo-id'], ['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo-id', 'content\n')
        tree.commit('one', rev_id='rev-one')
        tree.rename_one('foo', 'bar')
        self.assertEqual('bar', tree.id2path('foo-id'))
        self.assertEqual('content\n', tree._file_transport.get_bytes('bar'))
        self.assertRaises(errors.NoSuchFile, tree._file_transport.get_bytes, 'foo')
        tree.commit('two', rev_id='rev-two')
        self.assertEqual('content\n', tree._file_transport.get_bytes('bar'))
        self.assertRaises(errors.NoSuchFile, tree._file_transport.get_bytes, 'foo')
        rev_tree2 = tree.branch.repository.revision_tree('rev-two')
        self.assertEqual('bar', rev_tree2.id2path('foo-id'))
        self.assertEqual('content\n', rev_tree2.get_file_text('foo-id'))

    def test_rename_file_to_subdir(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_memory_tree('branch')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add('')
        tree.mkdir('subdir', 'subdir-id')
        tree.add('foo', 'foo-id', 'file')
        tree.put_file_bytes_non_atomic('foo-id', 'content\n')
        tree.commit('one', rev_id='rev-one')
        tree.rename_one('foo', 'subdir/bar')
        self.assertEqual('subdir/bar', tree.id2path('foo-id'))
        self.assertEqual('content\n', tree._file_transport.get_bytes('subdir/bar'))
        tree.commit('two', rev_id='rev-two')
        rev_tree2 = tree.branch.repository.revision_tree('rev-two')
        self.assertEqual('subdir/bar', rev_tree2.id2path('foo-id'))