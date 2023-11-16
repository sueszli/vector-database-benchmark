"""Tests for Branch.iter_merge_sorted_revisions()"""
from bzrlib import errors, revision, tests
from bzrlib.tests import per_branch

class TestIterMergeSortedRevisionsSimpleGraph(per_branch.TestCaseWithBranch):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestIterMergeSortedRevisionsSimpleGraph, self).setUp()
        builder = self.make_builder_with_merges('.')
        self.branch = builder.get_branch()
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)

    def make_builder_with_merges(self, relpath):
        if False:
            while True:
                i = 10
        try:
            builder = self.make_branch_builder(relpath)
        except (errors.TransportNotPossible, errors.UninitializableFormat):
            raise tests.TestNotApplicable('format not directly constructable')
        builder.start_series()
        builder.build_snapshot('1', None, [('add', ('', 'TREE_ROOT', 'directory', ''))])
        builder.build_snapshot('1.1.1', ['1'], [])
        builder.build_snapshot('2', ['1'], [])
        builder.build_snapshot('3', ['2', '1.1.1'], [])
        builder.finish_series()
        return builder

    def assertIterRevids(self, expected, *args, **kwargs):
        if False:
            return 10
        revids = [revid for (revid, depth, revno, eom) in self.branch.iter_merge_sorted_revisions(*args, **kwargs)]
        self.assertEqual(expected, revids)

    def test_merge_sorted(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['3', '1.1.1', '2', '1'])

    def test_merge_sorted_range(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['1.1.1'], start_revision_id='1.1.1', stop_revision_id='1')

    def test_merge_sorted_range_start_only(self):
        if False:
            while True:
                i = 10
        self.assertIterRevids(['1.1.1', '1'], start_revision_id='1.1.1')

    def test_merge_sorted_range_stop_exclude(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='1')

    def test_merge_sorted_range_stop_include(self):
        if False:
            return 10
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='include')

    def test_merge_sorted_range_stop_with_merges(self):
        if False:
            return 10
        self.assertIterRevids(['3', '1.1.1'], stop_revision_id='3', stop_rule='with-merges')

    def test_merge_sorted_range_stop_with_merges_can_show_non_parents(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='with-merges')

    def test_merge_sorted_range_stop_with_merges_ignore_non_parents(self):
        if False:
            print('Hello World!')
        self.assertIterRevids(['3', '1.1.1'], stop_revision_id='1.1.1', stop_rule='with-merges')

    def test_merge_sorted_single_stop_exclude(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids([], start_revision_id='3', stop_revision_id='3')

    def test_merge_sorted_single_stop_include(self):
        if False:
            return 10
        self.assertIterRevids(['3'], start_revision_id='3', stop_revision_id='3', stop_rule='include')

    def test_merge_sorted_single_stop_with_merges(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['3', '1.1.1'], start_revision_id='3', stop_revision_id='3', stop_rule='with-merges')

    def test_merge_sorted_forward(self):
        if False:
            while True:
                i = 10
        self.assertIterRevids(['1', '2', '1.1.1', '3'], direction='forward')

    def test_merge_sorted_range_forward(self):
        if False:
            print('Hello World!')
        self.assertIterRevids(['1.1.1'], start_revision_id='1.1.1', stop_revision_id='1', direction='forward')

    def test_merge_sorted_range_start_only_forward(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['1', '1.1.1'], start_revision_id='1.1.1', direction='forward')

    def test_merge_sorted_range_stop_exclude_forward(self):
        if False:
            i = 10
            return i + 15
        self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='1', direction='forward')

    def test_merge_sorted_range_stop_include_forward(self):
        if False:
            print('Hello World!')
        self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='2', stop_rule='include', direction='forward')

    def test_merge_sorted_range_stop_with_merges_forward(self):
        if False:
            while True:
                i = 10
        self.assertIterRevids(['1.1.1', '3'], stop_revision_id='3', stop_rule='with-merges', direction='forward')

class TestIterMergeSortedRevisionsBushyGraph(per_branch.TestCaseWithBranch):

    def make_branch_builder(self, relpath):
        if False:
            while True:
                i = 10
        try:
            builder = super(TestIterMergeSortedRevisionsBushyGraph, self).make_branch_builder(relpath)
        except (errors.TransportNotPossible, errors.UninitializableFormat):
            raise tests.TestNotApplicable('format not directly constructable')
        return builder

    def make_branch_with_embedded_merges(self, relpath='.'):
        if False:
            for i in range(10):
                print('nop')
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        builder.build_snapshot('1', None, [('add', ('', 'TREE_ROOT', 'directory', ''))])
        builder.build_snapshot('1.1.1', ['1'], [])
        builder.build_snapshot('2', ['1', '1.1.1'], [])
        builder.build_snapshot('2.1.1', ['2'], [])
        builder.build_snapshot('2.1.2', ['2.1.1'], [])
        builder.build_snapshot('2.2.1', ['2.1.1'], [])
        builder.build_snapshot('2.1.3', ['2.1.2', '2.2.1'], [])
        builder.build_snapshot('3', ['2'], [])
        builder.build_snapshot('4', ['3', '2.1.3'], [])
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def make_branch_with_different_depths_merges(self, relpath='.'):
        if False:
            i = 10
            return i + 15
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        builder.build_snapshot('1', None, [('add', ('', 'TREE_ROOT', 'directory', ''))])
        builder.build_snapshot('2', ['1'], [])
        builder.build_snapshot('1.1.1', ['1'], [])
        builder.build_snapshot('1.1.2', ['1.1.1'], [])
        builder.build_snapshot('1.2.1', ['1.1.1'], [])
        builder.build_snapshot('1.2.2', ['1.2.1'], [])
        builder.build_snapshot('1.3.1', ['1.2.1'], [])
        builder.build_snapshot('1.3.2', ['1.3.1'], [])
        builder.build_snapshot('1.4.1', ['1.3.1'], [])
        builder.build_snapshot('1.3.3', ['1.3.2', '1.4.11'], [])
        builder.build_snapshot('1.2.3', ['1.2.2', '1.3.3'], [])
        builder.build_snapshot('2.1.1', ['2'], [])
        builder.build_snapshot('2.1.2', ['2.1.1'], [])
        builder.build_snapshot('2.2.1', ['2.1.1'], [])
        builder.build_snapshot('2.1.3', ['2.1.2', '2.2.1'], [])
        builder.build_snapshot('3', ['2', '1.2.3'], [])
        builder.build_snapshot('4', ['3', '2.1.3'], [])
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def make_branch_with_alternate_ancestries(self, relpath='.'):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        builder.build_snapshot('1', None, [('add', ('', 'TREE_ROOT', 'directory', ''))])
        builder.build_snapshot('1.1.1', ['1'], [])
        builder.build_snapshot('2', ['1', '1.1.1'], [])
        builder.build_snapshot('1.2.1', ['1.1.1'], [])
        builder.build_snapshot('1.1.2', ['1.1.1', '1.2.1'], [])
        builder.build_snapshot('3', ['2', '1.1.2'], [])
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def assertIterRevids(self, expected, branch, *args, **kwargs):
        if False:
            return 10
        revs = list(branch.iter_merge_sorted_revisions(*args, **kwargs))
        revids = [revid for (revid, depth, revno, eom) in revs]
        self.assertEqual(expected, revids)

    def test_merge_sorted_starting_at_embedded_merge(self):
        if False:
            print('Hello World!')
        branch = self.make_branch_with_embedded_merges()
        self.assertIterRevids(['4', '2.1.3', '2.2.1', '2.1.2', '2.1.1', '3', '2', '1.1.1', '1'], branch)
        self.assertIterRevids(['2.2.1', '2.1.1', '2', '1.1.1', '1'], branch, start_revision_id='2.2.1', stop_rule='with-merges')

    def test_merge_sorted_with_different_depths_merge(self):
        if False:
            print('Hello World!')
        branch = self.make_branch_with_different_depths_merges()
        self.assertIterRevids(['4', '2.1.3', '2.2.1', '2.1.2', '2.1.1', '3', '1.2.3', '1.3.3', '1.3.2', '1.3.1', '1.2.2', '1.2.1', '1.1.1', '2', '1'], branch)
        self.assertIterRevids(['2.2.1', '2.1.1', '2', '1'], branch, start_revision_id='2.2.1', stop_rule='with-merges')

    def test_merge_sorted_exclude_ancestry(self):
        if False:
            i = 10
            return i + 15
        branch = self.make_branch_with_alternate_ancestries()
        self.assertIterRevids(['3', '1.1.2', '1.2.1', '2', '1.1.1', '1'], branch)
        self.assertIterRevids(['1.1.2', '1.2.1'], branch, stop_rule='with-merges-without-common-ancestry', start_revision_id='1.1.2', stop_revision_id='1.1.1')