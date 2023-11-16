"""Black-box tests for bzr verify-signatures."""
from bzrlib import gpg, tests
from bzrlib.tests.matchers import ContainsNoVfsCalls

class TestVerifySignatures(tests.TestCaseWithTransport):

    def monkey_patch_gpg(self):
        if False:
            i = 10
            return i + 15
        'Monkey patch the gpg signing strategy to be a loopback.\n\n        This also registers the cleanup, so that we will revert to\n        the original gpg strategy when done.\n        '
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def setup_tree(self, location='.'):
        if False:
            for i in range(10):
                print('nop')
        wt = self.make_branch_and_tree(location)
        wt.commit('base A', allow_pointless=True, rev_id='A')
        wt.commit('base B', allow_pointless=True, rev_id='B')
        wt.commit('base C', allow_pointless=True, rev_id='C')
        wt.commit('base D', allow_pointless=True, rev_id='D', committer='Alternate <alt@foo.com>')
        wt.add_parent_tree_id('aghost')
        wt.commit('base E', allow_pointless=True, rev_id='E')
        return wt

    def test_verify_signatures(self):
        if False:
            print('Hello World!')
        wt = self.setup_tree()
        self.monkey_patch_gpg()
        self.run_bzr('sign-my-commits')
        out = self.run_bzr('verify-signatures', retcode=1)
        self.assertEqual(('4 commits with valid signatures\n0 commits with key now expired\n0 commits with unknown keys\n0 commits not valid\n1 commit not signed\n', ''), out)

    def test_verify_signatures_acceptable_key(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.setup_tree()
        self.monkey_patch_gpg()
        self.run_bzr('sign-my-commits')
        out = self.run_bzr(['verify-signatures', '--acceptable-keys=foo,bar'], retcode=1)
        self.assertEqual(('4 commits with valid signatures\n0 commits with key now expired\n0 commits with unknown keys\n0 commits not valid\n1 commit not signed\n', ''), out)

    def test_verify_signatures_verbose(self):
        if False:
            i = 10
            return i + 15
        wt = self.setup_tree()
        self.monkey_patch_gpg()
        self.run_bzr('sign-my-commits')
        out = self.run_bzr('verify-signatures --verbose', retcode=1)
        self.assertEqual(('4 commits with valid signatures\n  None signed 4 commits\n0 commits with key now expired\n0 commits with unknown keys\n0 commits not valid\n1 commit not signed\n  1 commit by author Alternate <alt@foo.com>\n', ''), out)

    def test_verify_signatures_verbose_all_valid(self):
        if False:
            while True:
                i = 10
        wt = self.setup_tree()
        self.monkey_patch_gpg()
        self.run_bzr('sign-my-commits')
        self.run_bzr(['sign-my-commits', '.', 'Alternate <alt@foo.com>'])
        out = self.run_bzr('verify-signatures --verbose')
        self.assertEqual(('All commits signed with verifiable keys\n  None signed 5 commits\n', ''), out)

class TestSmartServerVerifySignatures(tests.TestCaseWithTransport):

    def monkey_patch_gpg(self):
        if False:
            i = 10
            return i + 15
        'Monkey patch the gpg signing strategy to be a loopback.\n\n        This also registers the cleanup, so that we will revert to\n        the original gpg strategy when done.\n        '
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def test_verify_signatures(self):
        if False:
            i = 10
            return i + 15
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', 'thecontents')])
        t.add('foo')
        t.commit('message')
        self.monkey_patch_gpg()
        (out, err) = self.run_bzr(['sign-my-commits', self.get_url('branch')])
        self.reset_smart_call_log()
        self.run_bzr('sign-my-commits')
        out = self.run_bzr(['verify-signatures', self.get_url('branch')])
        self.assertLength(10, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)