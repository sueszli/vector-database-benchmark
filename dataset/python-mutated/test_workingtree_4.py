"""Tests for WorkingTreeFormat4"""
import os
import time
from bzrlib import bzrdir, dirstate, errors, inventory, osutils, workingtree_4
from bzrlib.lockdir import LockDir
from bzrlib.tests import TestCaseWithTransport, TestSkipped, features
from bzrlib.tree import InterTree

class TestWorkingTreeFormat4(TestCaseWithTransport):
    """Tests specific to WorkingTreeFormat4."""

    def test_disk_layout(self):
        if False:
            while True:
                i = 10
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        control.create_repository()
        control.create_branch()
        tree = workingtree_4.WorkingTreeFormat4().initialize(control)
        t = control.get_workingtree_transport(None)
        self.assertEqualDiff('Bazaar Working Tree Format 4 (bzr 0.15)\n', t.get('format').read())
        self.assertFalse(t.has('inventory.basis'))
        self.assertFalse(t.has('last-revision'))
        state = dirstate.DirState.on_file(t.local_abspath('dirstate'))
        state.lock_read()
        try:
            self.assertEqual([], state.get_parent_ids())
        finally:
            state.unlock()

    def test_resets_ignores_on_last_unlock(self):
        if False:
            while True:
                i = 10
        tree = self.make_workingtree()
        tree.lock_read()
        try:
            tree.lock_read()
            try:
                tree.is_ignored('foo')
            finally:
                tree.unlock()
            self.assertIsNot(None, tree._ignoreglobster)
        finally:
            tree.unlock()
        self.assertIs(None, tree._ignoreglobster)

    def test_uses_lockdir(self):
        if False:
            i = 10
            return i + 15
        'WorkingTreeFormat4 uses its own LockDir:\n\n            - lock is a directory\n            - when the WorkingTree is locked, LockDir can see that\n        '
        t = self.get_transport()
        tree = self.make_workingtree()
        self.assertIsDirectory('.bzr', t)
        self.assertIsDirectory('.bzr/checkout', t)
        self.assertIsDirectory('.bzr/checkout/lock', t)
        our_lock = LockDir(t, '.bzr/checkout/lock')
        self.assertEqual(our_lock.peek(), None)
        tree.lock_write()
        self.assertTrue(our_lock.peek())
        tree.unlock()
        self.assertEqual(our_lock.peek(), None)

    def make_workingtree(self, relpath=''):
        if False:
            i = 10
            return i + 15
        url = self.get_url(relpath)
        if relpath:
            self.build_tree([relpath + '/'])
        dir = bzrdir.BzrDirMetaFormat1().initialize(url)
        repo = dir.create_repository()
        branch = dir.create_branch()
        try:
            return workingtree_4.WorkingTreeFormat4().initialize(dir)
        except errors.NotLocalUrl:
            raise TestSkipped('Not a local URL')

    def test_dirstate_stores_all_parent_inventories(self):
        if False:
            print('Hello World!')
        tree = self.make_workingtree()
        subtree = self.make_branch_and_tree('subdir')
        subtree.lock_write()
        self.addCleanup(subtree.unlock)
        self.build_tree(['subdir/file-a'])
        subtree.add(['file-a'], ['id-a'])
        rev1 = subtree.commit('commit in subdir')
        subtree2 = subtree.bzrdir.sprout('subdir2').open_workingtree()
        self.build_tree(['subdir2/file-b'])
        subtree2.add(['file-b'], ['id-b'])
        rev2 = subtree2.commit('commit in subdir2')
        subtree.flush()
        subtree3 = subtree.bzrdir.sprout('subdir3').open_workingtree()
        rev3 = subtree3.commit('merge from subdir2')
        repo = tree.branch.repository
        repo.fetch(subtree.branch.repository, rev1)
        repo.fetch(subtree2.branch.repository, rev2)
        repo.fetch(subtree3.branch.repository, rev3)
        rev1_revtree = repo.revision_tree(rev1)
        rev2_revtree = repo.revision_tree(rev2)
        rev3_revtree = repo.revision_tree(rev3)
        tree.set_parent_trees([(rev1, rev1_revtree), (rev2, rev2_revtree), (rev3, rev3_revtree)])
        rev1_tree = tree.revision_tree(rev1)
        rev1_tree.lock_read()
        self.addCleanup(rev1_tree.unlock)
        rev2_tree = tree.revision_tree(rev2)
        rev2_tree.lock_read()
        self.addCleanup(rev2_tree.unlock)
        rev3_tree = tree.revision_tree(rev3)
        rev3_tree.lock_read()
        self.addCleanup(rev3_tree.unlock)
        self.assertTreesEqual(rev1_revtree, rev1_tree)
        self.assertTreesEqual(rev2_revtree, rev2_tree)
        self.assertTreesEqual(rev3_revtree, rev3_tree)

    def test_dirstate_doesnt_read_parents_from_repo_when_setting(self):
        if False:
            for i in range(10):
                print('nop')
        "Setting parent trees on a dirstate working tree takes\n        the trees it's given and doesn't need to read them from the\n        repository.\n        "
        tree = self.make_workingtree()
        subtree = self.make_branch_and_tree('subdir')
        rev1 = subtree.commit('commit in subdir')
        rev1_tree = subtree.basis_tree()
        rev1_tree.lock_read()
        self.addCleanup(rev1_tree.unlock)
        tree.branch.pull(subtree.branch)
        repo = tree.branch.repository
        self.overrideAttr(repo, 'get_revision', self.fail)
        self.overrideAttr(repo, 'get_inventory', self.fail)
        self.overrideAttr(repo, '_get_inventory_xml', self.fail)
        tree.set_parent_trees([(rev1, rev1_tree)])

    def test_dirstate_doesnt_read_from_repo_when_returning_cache_tree(self):
        if False:
            return 10
        'Getting parent trees from a dirstate tree does not read from the\n        repos inventory store. This is an important part of the dirstate\n        performance optimisation work.\n        '
        tree = self.make_workingtree()
        subtree = self.make_branch_and_tree('subdir')
        subtree.lock_write()
        self.addCleanup(subtree.unlock)
        rev1 = subtree.commit('commit in subdir')
        rev1_tree = subtree.basis_tree()
        rev1_tree.lock_read()
        rev1_tree.root_inventory
        self.addCleanup(rev1_tree.unlock)
        rev2 = subtree.commit('second commit in subdir', allow_pointless=True)
        rev2_tree = subtree.basis_tree()
        rev2_tree.lock_read()
        rev2_tree.root_inventory
        self.addCleanup(rev2_tree.unlock)
        tree.branch.pull(subtree.branch)
        repo = tree.branch.repository
        self.overrideAttr(repo, 'get_inventory', self.fail)
        self.overrideAttr(repo, '_get_inventory_xml', self.fail)
        tree.set_parent_trees([(rev1, rev1_tree), (rev2, rev2_tree)])
        result_rev1_tree = tree.revision_tree(rev1)
        result_rev2_tree = tree.revision_tree(rev2)
        self.assertTreesEqual(rev1_tree, result_rev1_tree)
        self.assertTreesEqual(rev2_tree, result_rev2_tree)

    def test_dirstate_doesnt_cache_non_parent_trees(self):
        if False:
            for i in range(10):
                print('nop')
        'Getting parent trees from a dirstate tree does not read from the\n        repos inventory store. This is an important part of the dirstate\n        performance optimisation work.\n        '
        tree = self.make_workingtree()
        subtree = self.make_branch_and_tree('subdir')
        rev1 = subtree.commit('commit in subdir')
        tree.branch.pull(subtree.branch)
        self.assertRaises(errors.NoSuchRevision, tree.revision_tree, rev1)

    def test_no_dirstate_outside_lock(self):
        if False:
            return 10
        'Getting a dirstate object fails if there is no lock.'

        def lock_and_call_current_dirstate(tree, lock_method):
            if False:
                while True:
                    i = 10
            getattr(tree, lock_method)()
            tree.current_dirstate()
            tree.unlock()
        tree = self.make_workingtree()
        self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
        lock_and_call_current_dirstate(tree, 'lock_read')
        self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
        lock_and_call_current_dirstate(tree, 'lock_write')
        self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
        lock_and_call_current_dirstate(tree, 'lock_tree_write')
        self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)

    def test_set_parent_trees_uses_update_basis_by_delta(self):
        if False:
            return 10
        builder = self.make_branch_builder('source')
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot('A', [], [('add', ('', 'root-id', 'directory', None)), ('add', ('a', 'a-id', 'file', 'content\n'))])
        builder.build_snapshot('B', ['A'], [('modify', ('a-id', 'new content\nfor a\n')), ('add', ('b', 'b-id', 'file', 'b-content\n'))])
        tree = self.make_workingtree('tree')
        source_branch = builder.get_branch()
        tree.branch.repository.fetch(source_branch.repository, 'B')
        tree.pull(source_branch, stop_revision='A')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        state = tree.current_dirstate()
        called = []
        orig_update = state.update_basis_by_delta

        def log_update_basis_by_delta(delta, new_revid):
            if False:
                return 10
            called.append(new_revid)
            return orig_update(delta, new_revid)
        state.update_basis_by_delta = log_update_basis_by_delta
        basis = tree.basis_tree()
        self.assertEqual('a-id', basis.path2id('a'))
        self.assertEqual(None, basis.path2id('b'))

        def fail_set_parent_trees(trees, ghosts):
            if False:
                return 10
            raise AssertionError('dirstate.set_parent_trees() was called')
        state.set_parent_trees = fail_set_parent_trees
        repo = tree.branch.repository
        tree.pull(source_branch, stop_revision='B')
        self.assertEqual(['B'], called)
        basis = tree.basis_tree()
        self.assertEqual('a-id', basis.path2id('a'))
        self.assertEqual('b-id', basis.path2id('b'))

    def test_set_parent_trees_handles_missing_basis(self):
        if False:
            for i in range(10):
                print('nop')
        builder = self.make_branch_builder('source')
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot('A', [], [('add', ('', 'root-id', 'directory', None)), ('add', ('a', 'a-id', 'file', 'content\n'))])
        builder.build_snapshot('B', ['A'], [('modify', ('a-id', 'new content\nfor a\n')), ('add', ('b', 'b-id', 'file', 'b-content\n'))])
        builder.build_snapshot('C', ['A'], [('add', ('c', 'c-id', 'file', 'c-content\n'))])
        b_c = self.make_branch('branch_with_c')
        b_c.pull(builder.get_branch(), stop_revision='C')
        b_b = self.make_branch('branch_with_b')
        b_b.pull(builder.get_branch(), stop_revision='B')
        wt = b_b.create_checkout('tree', lightweight=True)
        fmt = wt.bzrdir.find_branch_format()
        fmt.set_reference(wt.bzrdir, None, b_c)
        wt = wt.bzrdir.open_workingtree()
        wt.set_parent_trees([('C', b_c.repository.revision_tree('C'))])
        self.assertEqual(None, wt.basis_tree().path2id('b'))

    def test_new_dirstate_on_new_lock(self):
        if False:
            return 10
        known_dirstates = set()

        def lock_and_compare_all_current_dirstate(tree, lock_method):
            if False:
                print('Hello World!')
            getattr(tree, lock_method)()
            state = tree.current_dirstate()
            self.assertFalse(state in known_dirstates)
            known_dirstates.add(state)
            tree.unlock()
        tree = self.make_workingtree()
        lock_and_compare_all_current_dirstate(tree, 'lock_read')
        lock_and_compare_all_current_dirstate(tree, 'lock_read')
        lock_and_compare_all_current_dirstate(tree, 'lock_tree_write')
        lock_and_compare_all_current_dirstate(tree, 'lock_tree_write')
        lock_and_compare_all_current_dirstate(tree, 'lock_write')
        lock_and_compare_all_current_dirstate(tree, 'lock_write')

    def test_constructing_invalid_interdirstate_raises(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_workingtree()
        rev_id = tree.commit('first post')
        rev_id2 = tree.commit('second post')
        rev_tree = tree.branch.repository.revision_tree(rev_id)
        self.assertRaises(Exception, workingtree_4.InterDirStateTree, rev_tree, tree)
        self.assertRaises(Exception, workingtree_4.InterDirStateTree, tree, rev_tree)

    def test_revtree_to_revtree_not_interdirstate(self):
        if False:
            print('Hello World!')
        tree = self.make_workingtree()
        rev_id = tree.commit('first post')
        rev_id2 = tree.commit('second post')
        rev_tree = tree.branch.repository.revision_tree(rev_id)
        rev_tree2 = tree.branch.repository.revision_tree(rev_id2)
        optimiser = InterTree.get(rev_tree, rev_tree2)
        self.assertIsInstance(optimiser, InterTree)
        self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))
        optimiser = InterTree.get(rev_tree2, rev_tree)
        self.assertIsInstance(optimiser, InterTree)
        self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))

    def test_revtree_not_in_dirstate_to_dirstate_not_interdirstate(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_workingtree()
        rev_id = tree.commit('first post')
        rev_id2 = tree.commit('second post')
        rev_tree = tree.branch.repository.revision_tree(rev_id)
        tree.lock_read()
        optimiser = InterTree.get(rev_tree, tree)
        self.assertIsInstance(optimiser, InterTree)
        self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))
        optimiser = InterTree.get(tree, rev_tree)
        self.assertIsInstance(optimiser, InterTree)
        self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))
        tree.unlock()

    def test_empty_basis_to_dirstate_tree(self):
        if False:
            while True:
                i = 10
        tree = self.make_workingtree()
        tree.lock_read()
        basis_tree = tree.basis_tree()
        basis_tree.lock_read()
        optimiser = InterTree.get(basis_tree, tree)
        tree.unlock()
        basis_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_nonempty_basis_to_dirstate_tree(self):
        if False:
            while True:
                i = 10
        tree = self.make_workingtree()
        tree.commit('first post')
        tree.lock_read()
        basis_tree = tree.basis_tree()
        basis_tree.lock_read()
        optimiser = InterTree.get(basis_tree, tree)
        tree.unlock()
        basis_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_empty_basis_revtree_to_dirstate_tree(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_workingtree()
        tree.lock_read()
        basis_tree = tree.branch.repository.revision_tree(tree.last_revision())
        basis_tree.lock_read()
        optimiser = InterTree.get(basis_tree, tree)
        tree.unlock()
        basis_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_nonempty_basis_revtree_to_dirstate_tree(self):
        if False:
            return 10
        tree = self.make_workingtree()
        tree.commit('first post')
        tree.lock_read()
        basis_tree = tree.branch.repository.revision_tree(tree.last_revision())
        basis_tree.lock_read()
        optimiser = InterTree.get(basis_tree, tree)
        tree.unlock()
        basis_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_tree_to_basis_in_other_tree(self):
        if False:
            return 10
        tree = self.make_workingtree('a')
        tree.commit('first post')
        tree2 = self.make_workingtree('b')
        tree2.pull(tree.branch)
        basis_tree = tree.basis_tree()
        tree2.lock_read()
        basis_tree.lock_read()
        optimiser = InterTree.get(basis_tree, tree2)
        tree2.unlock()
        basis_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_merged_revtree_to_tree(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_workingtree('a')
        tree.commit('first post')
        tree.commit('tree 1 commit 2')
        tree2 = self.make_workingtree('b')
        tree2.pull(tree.branch)
        tree2.commit('tree 2 commit 2')
        tree.merge_from_branch(tree2.branch)
        second_parent_tree = tree.revision_tree(tree.get_parent_ids()[1])
        second_parent_tree.lock_read()
        tree.lock_read()
        optimiser = InterTree.get(second_parent_tree, tree)
        tree.unlock()
        second_parent_tree.unlock()
        self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)

    def test_id2path(self):
        if False:
            print('Hello World!')
        tree = self.make_workingtree('tree')
        self.build_tree(['tree/a', 'tree/b'])
        tree.add(['a'], ['a-id'])
        self.assertEqual(u'a', tree.id2path('a-id'))
        self.assertRaises(errors.NoSuchId, tree.id2path, 'a')
        tree.commit('a')
        tree.add(['b'], ['b-id'])
        try:
            new_path = u'bμrry'
            tree.rename_one('a', new_path)
        except UnicodeEncodeError:
            new_path = 'c'
            tree.rename_one('a', new_path)
        self.assertEqual(new_path, tree.id2path('a-id'))
        tree.commit(u'bµrry')
        tree.unversion(['a-id'])
        self.assertRaises(errors.NoSuchId, tree.id2path, 'a-id')
        self.assertEqual('b', tree.id2path('b-id'))
        self.assertRaises(errors.NoSuchId, tree.id2path, 'c-id')

    def test_unique_root_id_per_tree(self):
        if False:
            while True:
                i = 10
        format_name = 'development-subtree'
        tree1 = self.make_branch_and_tree('tree1', format=format_name)
        tree2 = self.make_branch_and_tree('tree2', format=format_name)
        self.assertNotEqual(tree1.get_root_id(), tree2.get_root_id())
        rev1 = tree1.commit('first post')
        tree3 = tree1.bzrdir.sprout('tree3').open_workingtree()
        self.assertEqual(tree3.get_root_id(), tree1.get_root_id())

    def test_set_root_id(self):
        if False:
            return 10

        def validate():
            if False:
                print('Hello World!')
            wt.lock_read()
            try:
                wt.current_dirstate()._validate()
            finally:
                wt.unlock()
        wt = self.make_workingtree('tree')
        wt.set_root_id('TREE-ROOTID')
        validate()
        wt.commit('somenthing')
        validate()
        wt.set_root_id('tree-rootid')
        validate()
        wt.commit('again')
        validate()

    def test_default_root_id(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tag', format='dirstate-tags')
        self.assertEqual(inventory.ROOT_ID, tree.get_root_id())
        tree = self.make_branch_and_tree('subtree', format='development-subtree')
        self.assertNotEqual(inventory.ROOT_ID, tree.get_root_id())

    def test_non_subtree_with_nested_trees(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.', format='dirstate')
        self.assertFalse(tree.supports_tree_reference())
        self.build_tree(['dir/'])
        tree.set_root_id('root')
        tree.add(['dir'], ['dir-id'])
        subtree = self.make_branch_and_tree('dir')
        self.assertEqual('directory', tree.kind('dir-id'))
        tree.lock_read()
        expected = [('dir-id', (None, u'dir'), True, (False, True), (None, 'root'), (None, u'dir'), (None, 'directory'), (None, False)), ('root', (None, u''), True, (False, True), (None, None), (None, u''), (None, 'directory'), (None, 0))]
        self.assertEqual(expected, list(tree.iter_changes(tree.basis_tree(), specific_files=['dir'])))
        tree.unlock()
        tree.commit('first post')
        os.rename('dir', 'also-dir')
        tree.lock_read()
        expected = [('dir-id', (u'dir', u'dir'), True, (True, True), ('root', 'root'), ('dir', 'dir'), ('directory', None), (False, False))]
        self.assertEqual(expected, list(tree.iter_changes(tree.basis_tree())))
        tree.unlock()

    def test_with_subtree_supports_tree_references(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.', format='development-subtree')
        self.assertTrue(tree.supports_tree_reference())

    def test_iter_changes_ignores_unversioned_dirs(self):
        if False:
            while True:
                i = 10
        'iter_changes should not descend into unversioned directories.'
        tree = self.make_branch_and_tree('.', format='dirstate')
        self.build_tree(['unversioned/', 'unversioned/a', 'unversioned/b/', 'versioned/', 'versioned/unversioned/', 'versioned/unversioned/a', 'versioned/unversioned/b/', 'versioned2/', 'versioned2/a', 'versioned2/unversioned/', 'versioned2/unversioned/a', 'versioned2/unversioned/b/'])
        tree.add(['versioned', 'versioned2', 'versioned2/a'])
        tree.commit('one', rev_id='rev-1')
        returned = []

        def walkdirs_spy(*args, **kwargs):
            if False:
                return 10
            for val in orig(*args, **kwargs):
                returned.append(val[0][0])
                yield val
        orig = self.overrideAttr(osutils, '_walkdirs_utf8', walkdirs_spy)
        basis = tree.basis_tree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        basis.lock_read()
        self.addCleanup(basis.unlock)
        changes = [c[1] for c in tree.iter_changes(basis, want_unversioned=True)]
        self.assertEqual([(None, 'unversioned'), (None, 'versioned/unversioned'), (None, 'versioned2/unversioned')], changes)
        self.assertEqual(['', 'versioned', 'versioned2'], returned)
        del returned[:]
        changes = [c[1] for c in tree.iter_changes(basis)]
        self.assertEqual([], changes)
        self.assertEqual(['', 'versioned', 'versioned2'], returned)

    def test_iter_changes_unversioned_error(self):
        if False:
            while True:
                i = 10
        ' Check if a PathsNotVersionedError is correctly raised and the\n            paths list contains all unversioned entries only.\n        '
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/bar', '')])
        tree.add(['bar'], ['bar-id'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        tree_iter_changes = lambda files: [c for c in tree.iter_changes(tree.basis_tree(), specific_files=files, require_versioned=True)]
        e = self.assertRaises(errors.PathsNotVersionedError, tree_iter_changes, ['bar', 'foo'])
        self.assertEqual(e.paths, ['foo'])

    def test_iter_changes_unversioned_non_ascii(self):
        if False:
            return 10
        'Unversioned non-ascii paths should be reported as unicode'
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('f', '')])
        tree.add(['f'], ['f-id'])

        def tree_iter_changes(tree, files):
            if False:
                return 10
            return list(tree.iter_changes(tree.basis_tree(), specific_files=files, require_versioned=True))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        e = self.assertRaises(errors.PathsNotVersionedError, tree_iter_changes, tree, [u'§', u'π'])
        self.assertEqual(e.paths, [u'§', u'π'])

    def get_tree_with_cachable_file_foo(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree_contents([('foo', 'a bit of content for foo\n')])
        tree.add(['foo'], ['foo-id'])
        tree.current_dirstate()._cutoff_time = time.time() + 60
        return tree

    def test_commit_updates_hash_cache(self):
        if False:
            print('Hello World!')
        tree = self.get_tree_with_cachable_file_foo()
        revid = tree.commit('a commit')
        entry = tree._get_entry(path='foo')
        expected_sha1 = osutils.sha_file_by_name('foo')
        self.assertEqual(expected_sha1, entry[1][0][1])
        self.assertEqual(len('a bit of content for foo\n'), entry[1][0][2])

    def test_observed_sha1_cachable(self):
        if False:
            while True:
                i = 10
        tree = self.get_tree_with_cachable_file_foo()
        expected_sha1 = osutils.sha_file_by_name('foo')
        statvalue = os.lstat('foo')
        tree._observed_sha1('foo-id', 'foo', (expected_sha1, statvalue))
        entry = tree._get_entry(path='foo')
        entry_state = entry[1][0]
        self.assertEqual(expected_sha1, entry_state[1])
        self.assertEqual(statvalue.st_size, entry_state[2])
        tree.unlock()
        tree.lock_read()
        tree = tree.bzrdir.open_workingtree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        entry = tree._get_entry(path='foo')
        entry_state = entry[1][0]
        self.assertEqual(expected_sha1, entry_state[1])
        self.assertEqual(statvalue.st_size, entry_state[2])

    def test_observed_sha1_new_file(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        tree.add(['foo'], ['foo-id'])
        tree.lock_read()
        try:
            current_sha1 = tree._get_entry(path='foo')[1][0][1]
        finally:
            tree.unlock()
        tree.lock_write()
        try:
            tree._observed_sha1('foo-id', 'foo', (osutils.sha_file_by_name('foo'), os.lstat('foo')))
            self.assertEqual(current_sha1, tree._get_entry(path='foo')[1][0][1])
        finally:
            tree.unlock()

    def test_get_file_with_stat_id_only(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        tree.add(['foo'], ['foo-id'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        (file_obj, statvalue) = tree.get_file_with_stat('foo-id')
        expected = os.lstat('foo')
        self.assertEqualStat(expected, statvalue)
        self.assertEqual(['contents of foo\n'], file_obj.readlines())

class TestCorruptDirstate(TestCaseWithTransport):
    """Tests for how we handle when the dirstate has been corrupted."""

    def create_wt4(self):
        if False:
            print('Hello World!')
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        control.create_repository()
        control.create_branch()
        tree = workingtree_4.WorkingTreeFormat4().initialize(control)
        return tree

    def test_invalid_rename(self):
        if False:
            i = 10
            return i + 15
        tree = self.create_wt4()
        tree.lock_write()
        try:
            tree.commit('init')
            state = tree.current_dirstate()
            state._read_dirblocks_if_needed()
            state._dirblocks[1][1].append((('', 'foo', 'foo-id'), [('f', '', 0, False, ''), ('r', 'bar', 0, False, '')]))
            self.assertListRaises(errors.CorruptDirstate, tree.iter_changes, tree.basis_tree())
        finally:
            tree.unlock()

    def get_simple_dirblocks(self, state):
        if False:
            while True:
                i = 10
        'Extract the simple information from the DirState.\n\n        This returns the dirblocks, only with the sha1sum and stat details\n        filtered out.\n        '
        simple_blocks = []
        for block in state._dirblocks:
            simple_block = (block[0], [])
            for entry in block[1]:
                simple_block[1].append((entry[0], [i[0] for i in entry[1]]))
            simple_blocks.append(simple_block)
        return simple_blocks

    def test_update_basis_with_invalid_delta(self):
        if False:
            i = 10
            return i + 15
        'When given an invalid delta, it should abort, and not be saved.'
        self.build_tree(['dir/', 'dir/file'])
        tree = self.create_wt4()
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add(['dir', 'dir/file'], ['dir-id', 'file-id'])
        first_revision_id = tree.commit('init')
        root_id = tree.path2id('')
        state = tree.current_dirstate()
        state._read_dirblocks_if_needed()
        self.assertEqual([('', [(('', '', root_id), ['d', 'd'])]), ('', [(('', 'dir', 'dir-id'), ['d', 'd'])]), ('dir', [(('dir', 'file', 'file-id'), ['f', 'f'])])], self.get_simple_dirblocks(state))
        tree.remove(['dir/file'])
        self.assertEqual([('', [(('', '', root_id), ['d', 'd'])]), ('', [(('', 'dir', 'dir-id'), ['d', 'd'])]), ('dir', [(('dir', 'file', 'file-id'), ['a', 'f'])])], self.get_simple_dirblocks(state))
        tree.flush()
        new_dir = inventory.InventoryDirectory('dir-id', 'new-dir', root_id)
        new_dir.revision = 'new-revision-id'
        new_file = inventory.InventoryFile('file-id', 'new-file', root_id)
        new_file.revision = 'new-revision-id'
        self.assertRaises(errors.InconsistentDelta, tree.update_basis_by_delta, 'new-revision-id', [('dir', 'new-dir', 'dir-id', new_dir), ('dir/file', 'new-dir/new-file', 'file-id', new_file)])
        del state
        tree.unlock()
        tree.lock_read()
        self.assertEqual(first_revision_id, tree.last_revision())
        state = tree.current_dirstate()
        state._read_dirblocks_if_needed()
        self.assertEqual([('', [(('', '', root_id), ['d', 'd'])]), ('', [(('', 'dir', 'dir-id'), ['d', 'd'])]), ('dir', [(('dir', 'file', 'file-id'), ['a', 'f'])])], self.get_simple_dirblocks(state))

class TestInventoryCoherency(TestCaseWithTransport):

    def test_inventory_is_synced_when_unversioning_a_dir(self):
        if False:
            return 10
        'Unversioning the root of a subtree unversions the entire subtree.'
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'c/'])
        tree.add(['a', 'a/b', 'c'], ['a-id', 'b-id', 'c-id'])
        tree.lock_write()
        self.addCleanup(tree.unlock)
        inv = tree.root_inventory
        self.assertTrue(inv.has_id('a-id'))
        self.assertTrue(inv.has_id('b-id'))
        tree.unversion(['a-id', 'b-id'])
        self.assertFalse(inv.has_id('a-id'))
        self.assertFalse(inv.has_id('b-id'))