"""Tests for Branch.sprout()"""
import os
from bzrlib import branch as _mod_branch, errors, osutils, remote, revision as _mod_revision, tests
from bzrlib.tests import features
from bzrlib.tests.per_branch import TestCaseWithBranch

class TestSprout(TestCaseWithBranch):

    def test_sprout_branch_nickname(self):
        if False:
            while True:
                i = 10
        raise tests.TestSkipped('XXX branch sprouting is not yet tested.')

    def test_sprout_branch_parent(self):
        if False:
            print('Hello World!')
        source = self.make_branch('source')
        target = source.bzrdir.sprout(self.get_url('target')).open_branch()
        self.assertEqual(source.bzrdir.root_transport.base, target.get_parent())

    def test_sprout_uses_bzrdir_branch_format(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.branch_format, _mod_branch.BranchReferenceFormat):
            raise tests.TestNotApplicable('cannot sprout to a reference')
        source = tests.TestCaseWithTransport.make_branch(self, 'old-branch', format='knit')
        target_bzrdir = self.make_bzrdir('target')
        target_bzrdir.create_repository()
        result_format = self.branch_format
        if isinstance(target_bzrdir, remote.RemoteBzrDir):
            target_bzrdir._format.set_branch_format(_mod_branch.BzrBranchFormat6())
            result_format = target_bzrdir._format.get_branch_format()
        target = source.sprout(target_bzrdir)
        if isinstance(target, remote.RemoteBranch):
            target._ensure_real()
            target = target._real_branch
        if isinstance(result_format, remote.RemoteBranchFormat):
            result_format = result_format._custom_format
        self.assertIs(result_format.__class__, target._format.__class__)

    def test_sprout_partial(self):
        if False:
            for i in range(10):
                print('nop')
        wt_a = self.make_branch_and_tree('a')
        self.build_tree(['a/one'])
        wt_a.add(['one'])
        wt_a.commit('commit one', rev_id='1')
        self.build_tree(['a/two'])
        wt_a.add(['two'])
        wt_a.commit('commit two', rev_id='2')
        repo_b = self.make_repository('b')
        repo_a = wt_a.branch.repository
        repo_a.copy_content_into(repo_b)
        br_b = wt_a.branch.sprout(repo_b.bzrdir, revision_id='1')
        self.assertEqual('1', br_b.last_revision())

    def test_sprout_partial_not_in_revision_history(self):
        if False:
            for i in range(10):
                print('nop')
        'We should be able to sprout from any revision in ancestry.'
        wt = self.make_branch_and_tree('source')
        self.build_tree(['source/a'])
        wt.add('a')
        wt.commit('rev1', rev_id='rev1')
        wt.commit('rev2-alt', rev_id='rev2-alt')
        wt.set_parent_ids(['rev1'])
        wt.branch.set_last_revision_info(1, 'rev1')
        wt.commit('rev2', rev_id='rev2')
        wt.set_parent_ids(['rev2', 'rev2-alt'])
        wt.commit('rev3', rev_id='rev3')
        repo = self.make_repository('target')
        repo.fetch(wt.branch.repository)
        branch2 = wt.branch.sprout(repo.bzrdir, revision_id='rev2-alt')
        self.assertEqual((2, 'rev2-alt'), branch2.last_revision_info())
        self.assertEqual('rev2-alt', branch2.last_revision())

    def test_sprout_preserves_tags(self):
        if False:
            while True:
                i = 10
        'Sprout preserves tags, even tags of absent revisions.'
        try:
            builder = self.make_branch_builder('source')
        except errors.UninitializableFormat:
            raise tests.TestSkipped('Uninitializable branch format')
        builder.build_commit(message='Rev 1', rev_id='rev-1')
        source = builder.get_branch()
        try:
            source.tags.set_tag('tag-a', 'missing-rev')
        except (errors.TagsNotSupported, errors.GhostTagsNotSupported):
            raise tests.TestNotApplicable('Branch format does not support tags or tags to ghosts.')
        target_bzrdir = self.make_repository('target').bzrdir
        new_branch = source.sprout(target_bzrdir)
        self.assertEqual('missing-rev', new_branch.tags.lookup_tag('tag-a'))

    def test_sprout_from_any_repo_revision(self):
        if False:
            i = 10
            return i + 15
        'We should be able to sprout from any revision.'
        wt = self.make_branch_and_tree('source')
        self.build_tree(['source/a'])
        wt.add('a')
        wt.commit('rev1a', rev_id='rev1a')
        wt.branch.set_last_revision_info(0, _mod_revision.NULL_REVISION)
        wt.set_last_revision(_mod_revision.NULL_REVISION)
        wt.revert()
        wt.commit('rev1b', rev_id='rev1b')
        wt2 = wt.bzrdir.sprout('target', revision_id='rev1a').open_workingtree()
        self.assertEqual('rev1a', wt2.last_revision())
        self.assertPathExists('target/a')

    def test_sprout_with_unicode_symlink(self):
        if False:
            print('Hello World!')
        self.requireFeature(features.SymlinkFeature)
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree1')
        target = u'Ω'
        link_name = u'€link'
        os.symlink(target, 'tree1/' + link_name)
        tree.add([link_name], ['link-id'])
        revision = tree.commit('added a link to a Unicode target')
        tree.bzrdir.sprout('dest')
        self.assertEqual(target, osutils.readlink('dest/' + link_name))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(target, tree.get_symlink_target('link-id'))
        self.assertEqual(target, tree.basis_tree().get_symlink_target('link-id'))

    def test_sprout_with_ghost_in_mainline(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree1')
        if not tree.branch.repository._format.supports_ghosts:
            raise tests.TestNotApplicable('repository format does not support ghosts in mainline')
        tree.set_parent_ids(['spooky'], allow_leftmost_as_ghost=True)
        tree.add('')
        tree.commit('msg1', rev_id='rev1')
        tree.commit('msg2', rev_id='rev2')
        tree.bzrdir.sprout('target', revision_id='rev1')

    def assertBranchHookBranchIsStacked(self, pre_change_params):
        if False:
            print('Hello World!')
        pre_change_params.branch.get_stacked_on_url()
        self.hook_calls.append(pre_change_params)

    def test_sprout_stacked_hooks_get_stacked_branch(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('source')
        tree.commit('a commit')
        revid = tree.commit('a second commit')
        source = tree.branch
        target_transport = self.get_transport('target')
        self.hook_calls = []
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', self.assertBranchHookBranchIsStacked, None)
        try:
            dir = source.bzrdir.sprout(target_transport.base, source.last_revision(), possible_transports=[target_transport], source_branch=source, stacked=True)
        except errors.UnstackableBranchFormat:
            if not self.branch_format.supports_stacking():
                raise tests.TestNotApplicable("Format doesn't auto stack successfully.")
            else:
                raise
        result = dir.open_branch()
        self.assertEqual(revid, result.last_revision())
        self.assertEqual(source.base, result.get_stacked_on_url())
        if isinstance(result, remote.RemoteBranch):
            expected_calls = 2
        else:
            expected_calls = 1
        self.assertEqual(expected_calls, len(self.hook_calls))