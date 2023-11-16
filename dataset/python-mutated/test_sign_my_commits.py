"""Black-box tests for bzr sign-my-commits."""
from bzrlib import gpg, tests
from bzrlib.tests.matchers import ContainsNoVfsCalls

class SignMyCommits(tests.TestCaseWithTransport):

    def monkey_patch_gpg(self):
        if False:
            i = 10
            return i + 15
        'Monkey patch the gpg signing strategy to be a loopback.\n\n        This also registers the cleanup, so that we will revert to\n        the original gpg strategy when done.\n        '
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def setup_tree(self, location='.'):
        if False:
            i = 10
            return i + 15
        wt = self.make_branch_and_tree(location)
        wt.commit('base A', allow_pointless=True, rev_id='A')
        wt.commit('base B', allow_pointless=True, rev_id='B')
        wt.commit('base C', allow_pointless=True, rev_id='C')
        wt.commit('base D', allow_pointless=True, rev_id='D', committer='Alternate <alt@foo.com>')
        wt.add_parent_tree_id('aghost')
        wt.commit('base E', allow_pointless=True, rev_id='E')
        return wt

    def assertUnsigned(self, repo, revision_id):
        if False:
            i = 10
            return i + 15
        'Assert that revision_id is not signed in repo.'
        self.assertFalse(repo.has_signature_for_revision_id(revision_id))

    def assertSigned(self, repo, revision_id):
        if False:
            return 10
        'Assert that revision_id is signed in repo.'
        self.assertTrue(repo.has_signature_for_revision_id(revision_id))

    def test_sign_my_commits(self):
        if False:
            return 10
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.assertUnsigned(repo, 'A')
        self.assertUnsigned(repo, 'B')
        self.assertUnsigned(repo, 'C')
        self.assertUnsigned(repo, 'D')
        self.run_bzr('sign-my-commits')
        self.assertSigned(repo, 'A')
        self.assertSigned(repo, 'B')
        self.assertSigned(repo, 'C')
        self.assertUnsigned(repo, 'D')

    def test_sign_my_commits_location(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.setup_tree('other')
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('sign-my-commits other')
        self.assertSigned(repo, 'A')
        self.assertSigned(repo, 'B')
        self.assertSigned(repo, 'C')
        self.assertUnsigned(repo, 'D')

    def test_sign_diff_committer(self):
        if False:
            i = 10
            return i + 15
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr(['sign-my-commits', '.', 'Alternate <alt@foo.com>'])
        self.assertUnsigned(repo, 'A')
        self.assertUnsigned(repo, 'B')
        self.assertUnsigned(repo, 'C')
        self.assertSigned(repo, 'D')

    def test_sign_dry_run(self):
        if False:
            print('Hello World!')
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        out = self.run_bzr('sign-my-commits --dry-run')[0]
        outlines = out.splitlines()
        self.assertEqual(5, len(outlines))
        self.assertEqual('Signed 4 revisions.', outlines[-1])
        self.assertUnsigned(repo, 'A')
        self.assertUnsigned(repo, 'B')
        self.assertUnsigned(repo, 'C')
        self.assertUnsigned(repo, 'D')
        self.assertUnsigned(repo, 'E')

class TestSmartServerSignMyCommits(tests.TestCaseWithTransport):

    def monkey_patch_gpg(self):
        if False:
            print('Hello World!')
        'Monkey patch the gpg signing strategy to be a loopback.\n\n        This also registers the cleanup, so that we will revert to\n        the original gpg strategy when done.\n        '
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def test_sign_single_commit(self):
        if False:
            print('Hello World!')
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', 'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        self.monkey_patch_gpg()
        (out, err) = self.run_bzr(['sign-my-commits', self.get_url('branch')])
        self.assertLength(15, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)