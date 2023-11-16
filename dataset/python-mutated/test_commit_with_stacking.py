from bzrlib import errors, remote, tests, urlutils
from bzrlib.tests.per_repository import TestCaseWithRepository

class TestCaseWithStackedTarget(TestCaseWithRepository):
    r1_key = ('rev1-id',)
    r2_key = ('rev2-id',)

    def make_stacked_target(self):
        if False:
            while True:
                i = 10
        base_tree = self.make_branch_and_tree('base')
        self.build_tree(['base/f1.txt'])
        base_tree.add(['f1.txt'], ['f1.txt-id'])
        base_tree.commit('initial', rev_id=self.r1_key[0])
        self.build_tree(['base/f2.txt'])
        base_tree.add(['f2.txt'], ['f2.txt-id'])
        base_tree.commit('base adds f2', rev_id=self.r2_key[0])
        stacked_url = urlutils.join(base_tree.branch.base, '../stacked')
        stacked_bzrdir = base_tree.bzrdir.sprout(stacked_url, stacked=True)
        if isinstance(stacked_bzrdir, remote.RemoteBzrDir):
            stacked_branch = stacked_bzrdir.open_branch()
            stacked_tree = stacked_branch.create_checkout('stacked', lightweight=True)
        else:
            stacked_tree = stacked_bzrdir.open_workingtree()
        return (base_tree, stacked_tree)

class TestCommitWithStacking(TestCaseWithStackedTarget):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestCommitWithStacking, self).setUp()
        format = self.repository_format
        if not (isinstance(format, remote.RemoteRepositoryFormat) or format.supports_chks):
            raise tests.TestNotApplicable('stacked commit only supported for chk repositories')

    def get_only_repo(self, tree):
        if False:
            i = 10
            return i + 15
        'Open just the repository used by this tree.\n\n        This returns a read locked Repository object without any stacking\n        fallbacks.\n        '
        repo = tree.branch.repository.bzrdir.open_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        return repo

    def assertPresent(self, expected, vf, keys):
        if False:
            while True:
                i = 10
        'Check which of the supplied keys are present.'
        parent_map = vf.get_parent_map(keys)
        self.assertEqual(sorted(expected), sorted(parent_map))

    def test_simple_commit(self):
        if False:
            while True:
                i = 10
        (base_tree, stacked_tree) = self.make_stacked_target()
        self.assertEqual(1, len(stacked_tree.branch.repository._fallback_repositories))
        self.build_tree_contents([('stacked/f1.txt', 'new content\n')])
        stacked_tree.commit('new content', rev_id='new-rev-id')
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([self.r2_key], stacked_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_branch = base_tree.bzrdir.sprout('stacked2', stacked=True).open_branch()
        stacked2_branch.repository.fetch(stacked_only_repo, revision_id='new-rev-id')

    def test_merge_commit(self):
        if False:
            return 10
        (base_tree, stacked_tree) = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', 'new content\n')])
        r3_key = ('rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        to_be_merged_tree = base_tree.bzrdir.sprout('merged').open_workingtree()
        self.build_tree(['merged/f2.txt'])
        to_be_merged_tree.add(['f2.txt'], ['f2.txt-id'])
        to_merge_key = ('to-merge-rev-id',)
        to_be_merged_tree.commit('new-to-be-merged', rev_id=to_merge_key[0])
        stacked_tree.merge_from_branch(to_be_merged_tree.branch)
        merged_key = ('merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        all_keys = [self.r1_key, self.r2_key, r3_key, to_merge_key, merged_key]
        self.assertPresent([to_merge_key, merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, to_merge_key, merged_key], stacked_only_repo.inventories, all_keys)

    def test_merge_from_master(self):
        if False:
            i = 10
            return i + 15
        (base_tree, stacked_tree) = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', 'new content\n')])
        r3_key = ('rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        stacked_tree.merge_from_branch(base_tree.branch)
        merged_key = ('merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        all_keys = [self.r1_key, self.r2_key, r3_key, merged_key]
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, merged_key], stacked_only_repo.inventories, all_keys)

    def test_multi_stack(self):
        if False:
            print('Hello World!')
        'base + stacked + stacked-on-stacked'
        (base_tree, stacked_tree) = self.make_stacked_target()
        self.build_tree(['stacked/f3.txt'])
        stacked_tree.add(['f3.txt'], ['f3.txt-id'])
        stacked_key = ('stacked-rev-id',)
        stacked_tree.commit('add f3', rev_id=stacked_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([self.r2_key], stacked_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_url = urlutils.join(base_tree.branch.base, '../stacked2')
        stacked2_bzrdir = stacked_tree.bzrdir.sprout(stacked2_url, revision_id=self.r1_key[0], stacked=True)
        if isinstance(stacked2_bzrdir, remote.RemoteBzrDir):
            stacked2_branch = stacked2_bzrdir.open_branch()
            stacked2_tree = stacked2_branch.create_checkout('stacked2', lightweight=True)
        else:
            stacked2_tree = stacked2_bzrdir.open_workingtree()
        self.build_tree(['stacked2/f3.txt'])
        stacked2_only_repo = self.get_only_repo(stacked2_tree)
        self.assertPresent([], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_tree.add(['f3.txt'], ['f3.txt-id'])
        stacked2_tree.commit('add f3', rev_id='stacked2-rev-id')
        stacked2_only_repo.refresh_data()
        self.assertPresent([self.r1_key], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])

    def test_commit_with_ghosts_fails(self):
        if False:
            i = 10
            return i + 15
        (base_tree, stacked_tree) = self.make_stacked_target()
        stacked_tree.set_parent_ids([stacked_tree.last_revision(), 'ghost-rev-id'])
        self.assertRaises(errors.BzrError, stacked_tree.commit, 'failed_commit')

    def test_commit_with_ghost_in_ancestry(self):
        if False:
            while True:
                i = 10
        (base_tree, stacked_tree) = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', 'new content\n')])
        r3_key = ('rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        to_be_merged_tree = base_tree.bzrdir.sprout('merged').open_workingtree()
        self.build_tree(['merged/f2.txt'])
        to_be_merged_tree.add(['f2.txt'], ['f2.txt-id'])
        ghost_key = ('ghost-rev-id',)
        to_be_merged_tree.set_parent_ids([r3_key[0], ghost_key[0]])
        to_merge_key = ('to-merge-rev-id',)
        to_be_merged_tree.commit('new-to-be-merged', rev_id=to_merge_key[0])
        stacked_tree.merge_from_branch(to_be_merged_tree.branch)
        merged_key = ('merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        all_keys = [self.r1_key, self.r2_key, r3_key, to_merge_key, merged_key, ghost_key]
        self.assertPresent([to_merge_key, merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, to_merge_key, merged_key], stacked_only_repo.inventories, all_keys)

class TestCommitStackedFailsAppropriately(TestCaseWithStackedTarget):

    def test_stacked_commit_fails_on_old_formats(self):
        if False:
            for i in range(10):
                print('nop')
        (base_tree, stacked_tree) = self.make_stacked_target()
        format = stacked_tree.branch.repository._format
        if format.supports_chks:
            stacked_tree.commit('should succeed')
        else:
            self.assertRaises(errors.BzrError, stacked_tree.commit, 'unsupported format')