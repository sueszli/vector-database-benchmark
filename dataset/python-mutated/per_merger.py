"""Implementation tests for bzrlib.merge.Merger."""
import os
from bzrlib.conflicts import TextConflict
from bzrlib import errors, merge as _mod_merge
from bzrlib.tests import multiply_tests, TestCaseWithTransport
from bzrlib.tests.test_merge_core import MergeBuilder
from bzrlib.transform import TreeTransform

def load_tests(standard_tests, module, loader):
    if False:
        for i in range(10):
            print('nop')
    'Multiply tests for tranport implementations.'
    result = loader.suiteClass()
    scenarios = [(name, {'merge_type': merger}) for (name, merger) in _mod_merge.merge_type_registry.items()]
    return multiply_tests(standard_tests, scenarios, result)

class TestMergeImplementation(TestCaseWithTransport):

    def do_merge(self, target_tree, source_tree, **kwargs):
        if False:
            return 10
        merger = _mod_merge.Merger.from_revision_ids(None, target_tree, source_tree.last_revision(), other_branch=source_tree.branch)
        merger.merge_type = self.merge_type
        for (name, value) in kwargs.items():
            setattr(merger, name, value)
        merger.do_merge()

    def test_merge_specific_file(self):
        if False:
            i = 10
            return i + 15
        this_tree = self.make_branch_and_tree('this')
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        self.build_tree_contents([('this/file1', 'a\nb\n'), ('this/file2', 'a\nb\n')])
        this_tree.add(['file1', 'file2'])
        this_tree.commit('Added files')
        other_tree = this_tree.bzrdir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/file1', 'a\nb\nc\n'), ('other/file2', 'a\nb\nc\n')])
        other_tree.commit('modified both')
        self.build_tree_contents([('this/file1', 'd\na\nb\n'), ('this/file2', 'd\na\nb\n')])
        this_tree.commit('modified both')
        self.do_merge(this_tree, other_tree, interesting_files=['file1'])
        self.assertFileEqual('d\na\nb\nc\n', 'this/file1')
        self.assertFileEqual('d\na\nb\n', 'this/file2')

    def test_merge_move_and_change(self):
        if False:
            print('Hello World!')
        this_tree = self.make_branch_and_tree('this')
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        self.build_tree_contents([('this/file1', 'line 1\nline 2\nline 3\nline 4\n')])
        this_tree.add('file1')
        this_tree.commit('Added file')
        other_tree = this_tree.bzrdir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/file1', 'line 1\nline 2 to 2.1\nline 3\nline 4\n')])
        other_tree.commit('Changed 2 to 2.1')
        self.build_tree_contents([('this/file1', 'line 1\nline 3\nline 2\nline 4\n')])
        this_tree.commit('Swapped 2 & 3')
        self.do_merge(this_tree, other_tree)
        if self.merge_type is _mod_merge.LCAMerger:
            self.expectFailure("lca merge doesn't conflict for move and change", self.assertFileEqual, 'line 1\n<<<<<<< TREE\nline 3\nline 2\n=======\nline 2 to 2.1\nline 3\n>>>>>>> MERGE-SOURCE\nline 4\n', 'this/file1')
        else:
            self.assertFileEqual('line 1\n<<<<<<< TREE\nline 3\nline 2\n=======\nline 2 to 2.1\nline 3\n>>>>>>> MERGE-SOURCE\nline 4\n', 'this/file1')

    def test_modify_conflicts_with_delete(self):
        if False:
            return 10
        builder = self.make_branch_builder('test')
        builder.start_series()
        builder.build_snapshot('BASE-id', None, [('add', ('', None, 'directory', None)), ('add', ('foo', 'foo-id', 'file', 'a\nb\nc\nd\ne\n'))])
        builder.build_snapshot('OTHER-id', ['BASE-id'], [('modify', ('foo-id', 'a\nc\nd\ne\n'))])
        builder.build_snapshot('THIS-id', ['BASE-id'], [('modify', ('foo-id', 'a\nb2\nc\nd\nX\ne\n'))])
        builder.finish_series()
        branch = builder.get_branch()
        this_tree = branch.bzrdir.create_workingtree()
        this_tree.lock_write()
        self.addCleanup(this_tree.unlock)
        other_tree = this_tree.bzrdir.sprout('other', 'OTHER-id').open_workingtree()
        self.do_merge(this_tree, other_tree)
        if self.merge_type is _mod_merge.LCAMerger:
            self.expectFailure("lca merge doesn't track deleted lines", self.assertFileEqual, 'a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')
        else:
            self.assertFileEqual('a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')

    def get_limbodir_deletiondir(self, wt):
        if False:
            print('Hello World!')
        transform = TreeTransform(wt)
        limbodir = transform._limbodir
        deletiondir = transform._deletiondir
        transform.finalize()
        return (limbodir, deletiondir)

    def test_merge_with_existing_limbo_empty(self):
        if False:
            i = 10
            return i + 15
        'Empty limbo dir is just cleaned up - see bug 427773'
        wt = self.make_branch_and_tree('this')
        (limbodir, deletiondir) = self.get_limbodir_deletiondir(wt)
        os.mkdir(limbodir)
        self.do_merge(wt, wt)

    def test_merge_with_existing_limbo_non_empty(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_branch_and_tree('this')
        (limbodir, deletiondir) = self.get_limbodir_deletiondir(wt)
        os.mkdir(limbodir)
        os.mkdir(os.path.join(limbodir, 'something'))
        self.assertRaises(errors.ExistingLimbo, self.do_merge, wt, wt)
        self.assertRaises(errors.LockError, wt.unlock)

    def test_merge_with_pending_deletion_empty(self):
        if False:
            while True:
                i = 10
        wt = self.make_branch_and_tree('this')
        (limbodir, deletiondir) = self.get_limbodir_deletiondir(wt)
        os.mkdir(deletiondir)
        self.do_merge(wt, wt)

    def test_merge_with_pending_deletion_non_empty(self):
        if False:
            print('Hello World!')
        'Also see bug 427773'
        wt = self.make_branch_and_tree('this')
        (limbodir, deletiondir) = self.get_limbodir_deletiondir(wt)
        os.mkdir(deletiondir)
        os.mkdir(os.path.join(deletiondir, 'something'))
        self.assertRaises(errors.ExistingPendingDeletion, self.do_merge, wt, wt)
        self.assertRaises(errors.LockError, wt.unlock)

class TestHookMergeFileContent(TestCaseWithTransport):
    """Tests that the 'merge_file_content' hook is invoked."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestHookMergeFileContent, self).setUp()
        self.hook_log = []

    def install_hook_inactive(self):
        if False:
            while True:
                i = 10

        def inactive_factory(merger):
            if False:
                while True:
                    i = 10
            self.hook_log.append(('inactive',))
            return None
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', inactive_factory, 'test hook (inactive)')

    def install_hook_noop(self):
        if False:
            while True:
                i = 10
        test = self

        class HookNA(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                if False:
                    while True:
                        i = 10
                test.hook_log.append(('no-op',))
                return ('not_applicable', None)

        def hook_na_factory(merger):
            if False:
                print('Hello World!')
            return HookNA(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_na_factory, 'test hook (no-op)')

    def install_hook_success(self):
        if False:
            for i in range(10):
                print('nop')
        test = self

        class HookSuccess(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                if False:
                    print('Hello World!')
                test.hook_log.append(('success',))
                if merge_params.file_id == '1':
                    return ('success', ['text-merged-by-hook'])
                return ('not_applicable', None)

        def hook_success_factory(merger):
            if False:
                for i in range(10):
                    print('nop')
            return HookSuccess(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_success_factory, 'test hook (success)')

    def install_hook_conflict(self):
        if False:
            print('Hello World!')
        test = self

        class HookConflict(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                if False:
                    while True:
                        i = 10
                test.hook_log.append(('conflict',))
                if merge_params.file_id == '1':
                    return ('conflicted', ['text-with-conflict-markers-from-hook'])
                return ('not_applicable', None)

        def hook_conflict_factory(merger):
            if False:
                for i in range(10):
                    print('nop')
            return HookConflict(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_conflict_factory, 'test hook (delete)')

    def install_hook_delete(self):
        if False:
            return 10
        test = self

        class HookDelete(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                if False:
                    while True:
                        i = 10
                test.hook_log.append(('delete',))
                if merge_params.file_id == '1':
                    return ('delete', None)
                return ('not_applicable', None)

        def hook_delete_factory(merger):
            if False:
                i = 10
                return i + 15
            return HookDelete(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_delete_factory, 'test hook (delete)')

    def install_hook_log_lines(self):
        if False:
            i = 10
            return i + 15
        'Install a hook that saves the get_lines for the this, base and other\n        versions of the file.\n        '
        test = self

        class HookLogLines(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                if False:
                    for i in range(10):
                        print('nop')
                test.hook_log.append(('log_lines', merge_params.this_lines, merge_params.other_lines, merge_params.base_lines))
                return ('not_applicable', None)

        def hook_log_lines_factory(merger):
            if False:
                i = 10
                return i + 15
            return HookLogLines(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_log_lines_factory, 'test hook (log_lines)')

    def make_merge_builder(self):
        if False:
            return 10
        builder = MergeBuilder(self.test_base_dir)
        self.addCleanup(builder.cleanup)
        return builder

    def create_file_needing_contents_merge(self, builder, file_id):
        if False:
            return 10
        builder.add_file(file_id, builder.tree_root, 'name1', 'text1', True)
        builder.change_contents(file_id, other='text4', this='text3')

    def test_change_vs_change(self):
        if False:
            print('Hello World!')
        'Hook is used for (changed, changed)'
        self.install_hook_success()
        builder = self.make_merge_builder()
        builder.add_file('1', builder.tree_root, 'name1', 'text1', True)
        builder.change_contents('1', other='text4', this='text3')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertEqual(builder.this.get_file('1').read(), 'text-merged-by-hook')

    def test_change_vs_deleted(self):
        if False:
            return 10
        'Hook is used for (changed, deleted)'
        self.install_hook_success()
        builder = self.make_merge_builder()
        builder.add_file('1', builder.tree_root, 'name1', 'text1', True)
        builder.change_contents('1', this='text2')
        builder.remove_file('1', other=True)
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertEqual(builder.this.get_file('1').read(), 'text-merged-by-hook')

    def test_result_can_be_delete(self):
        if False:
            while True:
                i = 10
        "A hook's result can be the deletion of a file."
        self.install_hook_delete()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertRaises(errors.NoSuchId, builder.this.id2path, '1')
        self.assertEqual([], list(builder.this.list_files()))

    def test_result_can_be_conflict(self):
        if False:
            i = 10
            return i + 15
        "A hook's result can be a conflict."
        self.install_hook_conflict()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [TextConflict('name1', file_id='1')])
        self.assertEqual(builder.this.get_file('1').read(), 'text-with-conflict-markers-from-hook')

    def test_can_access_this_other_and_base_versions(self):
        if False:
            print('Hello World!')
        'The hook function can call params.merger.get_lines to access the\n        THIS/OTHER/BASE versions of the file.\n        '
        self.install_hook_log_lines()
        builder = self.make_merge_builder()
        builder.add_file('1', builder.tree_root, 'name1', 'text1', True)
        builder.change_contents('1', this='text2', other='text3')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('log_lines', ['text2'], ['text3'], ['text1'])], self.hook_log)

    def test_chain_when_not_active(self):
        if False:
            while True:
                i = 10
        'When a hook function returns None, merging still works.'
        self.install_hook_inactive()
        self.install_hook_success()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertEqual(builder.this.get_file('1').read(), 'text-merged-by-hook')
        self.assertEqual([('inactive',), ('success',)], self.hook_log)

    def test_chain_when_not_applicable(self):
        if False:
            print('Hello World!')
        'When a hook function returns not_applicable, the next function is\n        tried (when one exists).\n        '
        self.install_hook_noop()
        self.install_hook_success()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertEqual(builder.this.get_file('1').read(), 'text-merged-by-hook')
        self.assertEqual([('no-op',), ('success',)], self.hook_log)

    def test_chain_stops_after_success(self):
        if False:
            while True:
                i = 10
        'When a hook function returns success, no later functions are tried.\n        '
        self.install_hook_success()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('success',)], self.hook_log)

    def test_chain_stops_after_conflict(self):
        if False:
            while True:
                i = 10
        'When a hook function returns conflict, no later functions are tried.\n        '
        self.install_hook_conflict()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('conflict',)], self.hook_log)

    def test_chain_stops_after_delete(self):
        if False:
            while True:
                i = 10
        'When a hook function returns delete, no later functions are tried.\n        '
        self.install_hook_delete()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, '1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('delete',)], self.hook_log)