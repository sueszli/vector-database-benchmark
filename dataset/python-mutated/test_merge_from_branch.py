"""Tests for the WorkingTree.merge_from_branch api."""
import os
from bzrlib import conflicts, errors, merge
from bzrlib.tests import per_workingtree

class TestMergeFromBranch(per_workingtree.TestCaseWithWorkingTree):

    def create_two_trees_for_merging(self):
        if False:
            while True:
                i = 10
        'Create two trees that can be merged from.\n\n        This sets self.tree_from, self.first_rev, self.tree_to, self.second_rev\n        and self.to_second_rev.\n        '
        self.tree_from = self.make_branch_and_tree('from')
        self.first_rev = self.tree_from.commit('first post')
        self.tree_to = self.tree_from.bzrdir.sprout('to').open_workingtree()
        self.second_rev = self.tree_from.commit('second rev', allow_pointless=True)
        self.to_second_rev = self.tree_to.commit('second rev', allow_pointless=True)

    def test_smoking_merge(self):
        if False:
            print('Hello World!')
        'Smoke test of merge_from_branch.'
        self.create_two_trees_for_merging()
        self.tree_to.merge_from_branch(self.tree_from.branch)
        self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())

    def test_merge_to_revision(self):
        if False:
            for i in range(10):
                print('nop')
        'Merge from a branch to a revision that is not the tip.'
        self.create_two_trees_for_merging()
        self.third_rev = self.tree_from.commit('real_tip')
        self.tree_to.merge_from_branch(self.tree_from.branch, to_revision=self.second_rev)
        self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())

    def test_compare_after_merge(self):
        if False:
            print('Hello World!')
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', 'text-a')])
        tree_a.add('file')
        tree_a.commit('added file')
        tree_b = tree_a.bzrdir.sprout('tree_b').open_workingtree()
        os.unlink('tree_a/file')
        tree_a.commit('deleted file')
        self.build_tree_contents([('tree_b/file', 'text-b')])
        tree_b.commit('changed file')
        tree_a.merge_from_branch(tree_b.branch)
        tree_a.lock_read()
        self.addCleanup(tree_a.unlock)
        list(tree_a.iter_changes(tree_a.basis_tree()))

    def test_merge_empty(self):
        if False:
            i = 10
            return i + 15
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', 'text-a')])
        tree_a.add('file')
        tree_a.commit('added file')
        tree_b = self.make_branch_and_tree('treeb')
        self.assertRaises(errors.NoCommits, tree_a.merge_from_branch, tree_b.branch)
        tree_b.merge_from_branch(tree_a.branch)

    def test_merge_base(self):
        if False:
            i = 10
            return i + 15
        tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', 'text-a')])
        tree_a.add('file')
        tree_a.commit('added file', rev_id='rev_1')
        tree_b = tree_a.bzrdir.sprout('tree_b').open_workingtree()
        os.unlink('tree_a/file')
        tree_a.commit('deleted file')
        self.build_tree_contents([('tree_b/file', 'text-b')])
        tree_b.commit('changed file')
        self.assertRaises(errors.PointlessMerge, tree_a.merge_from_branch, tree_b.branch, from_revision=tree_b.branch.last_revision())
        tree_a.merge_from_branch(tree_b.branch, from_revision='rev_1')
        tree_a.lock_read()
        self.addCleanup(tree_a.unlock)
        changes = list(tree_a.iter_changes(tree_a.basis_tree()))
        self.assertEqual(1, len(changes))

    def test_merge_type(self):
        if False:
            i = 10
            return i + 15
        this = self.make_branch_and_tree('this')
        self.build_tree_contents([('this/foo', 'foo')])
        this.add('foo', 'foo-id')
        this.commit('added foo')
        other = this.bzrdir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/foo', 'bar')])
        other.commit('content -> bar')
        self.build_tree_contents([('this/foo', 'baz')])
        this.commit('content -> baz')

        class QuxMerge(merge.Merge3Merger):

            def text_merge(self, file_id, trans_id):
                if False:
                    print('Hello World!')
                self.tt.create_file('qux', trans_id)
        this.merge_from_branch(other.branch, merge_type=QuxMerge)
        self.assertEqual('qux', this.get_file_text('foo-id'))

class TestMergedBranch(per_workingtree.TestCaseWithWorkingTree):

    def make_inner_branch(self):
        if False:
            return 10
        bld_inner = self.make_branch_builder('inner')
        bld_inner.start_series()
        bld_inner.build_snapshot('1', None, [('add', ('', 'inner-root-id', 'directory', '')), ('add', ('dir', 'dir-id', 'directory', '')), ('add', ('dir/file1', 'file1-id', 'file', 'file1 content\n')), ('add', ('file3', 'file3-id', 'file', 'file3 content\n'))])
        bld_inner.build_snapshot('4', ['1'], [('add', ('file4', 'file4-id', 'file', 'file4 content\n'))])
        bld_inner.build_snapshot('5', ['4'], [('rename', ('file4', 'dir/file4'))])
        bld_inner.build_snapshot('3', ['1'], [('modify', ('file3-id', 'new file3 contents\n'))])
        bld_inner.build_snapshot('2', ['1'], [('add', ('dir/file2', 'file2-id', 'file', 'file2 content\n'))])
        bld_inner.finish_series()
        br = bld_inner.get_branch()
        return br

    def assertTreeLayout(self, expected, tree):
        if False:
            for i in range(10):
                print('nop')
        tree.lock_read()
        try:
            actual = [e[0] for e in tree.list_files()]
            actual = sorted(actual)
            self.assertEqual(expected, actual)
        finally:
            tree.unlock()

    def make_outer_tree(self):
        if False:
            i = 10
            return i + 15
        outer = self.make_branch_and_tree('outer')
        self.build_tree_contents([('outer/foo', 'foo')])
        outer.add('foo', 'foo-id')
        outer.commit('added foo')
        inner = self.make_inner_branch()
        outer.merge_from_branch(inner, to_revision='1', from_revision='null:')
        outer.set_root_id(outer.basis_tree().get_root_id())
        outer.commit('merge inner branch')
        outer.mkdir('dir-outer', 'dir-outer-id')
        outer.move(['dir', 'file3'], to_dir='dir-outer')
        outer.commit('rename imported dir and file3 to dir-outer')
        return (outer, inner)

    def test_file1_deleted_in_dir(self):
        if False:
            return 10
        (outer, inner) = self.make_outer_tree()
        outer.remove(['dir-outer/dir/file1'], keep_files=False)
        outer.commit('delete file1')
        outer.merge_from_branch(inner)
        outer.commit('merge the rest')
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file2', 'dir-outer/file3', 'foo'], outer)

    def test_file3_deleted_in_root(self):
        if False:
            return 10
        (outer, inner) = self.make_outer_tree()
        outer.remove(['dir-outer/file3'], keep_files=False)
        outer.commit('delete file3')
        outer.merge_from_branch(inner)
        outer.commit('merge the rest')
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/dir/file2', 'foo'], outer)

    def test_file3_in_root_conflicted(self):
        if False:
            for i in range(10):
                print('nop')
        (outer, inner) = self.make_outer_tree()
        outer.remove(['dir-outer/file3'], keep_files=False)
        outer.commit('delete file3')
        nb_conflicts = outer.merge_from_branch(inner, to_revision='3')
        self.assertEqual(4, nb_conflicts)
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'file3.BASE', 'file3.OTHER', 'foo'], outer)

    def test_file4_added_in_root(self):
        if False:
            print('Hello World!')
        (outer, inner) = self.make_outer_tree()
        nb_conflicts = outer.merge_from_branch(inner, to_revision='4')
        self.assertEqual(1, nb_conflicts)
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/file3', 'file4', 'foo'], outer)

    def test_file4_added_then_renamed(self):
        if False:
            while True:
                i = 10
        (outer, inner) = self.make_outer_tree()
        self.assertEqual(1, outer.merge_from_branch(inner, to_revision='4'))
        try:
            outer.set_conflicts(conflicts.ConflictList())
        except errors.UnsupportedOperation:
            pass
        outer.commit('added file4')
        nb_conflicts = outer.merge_from_branch(inner, to_revision='5')
        self.assertEqual(1, nb_conflicts)
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/dir/file4', 'dir-outer/file3', 'foo'], outer)