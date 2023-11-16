"""Black-box tests for bzr re-sign.
"""
from bzrlib import gpg, tests
from bzrlib.controldir import ControlDir
from bzrlib.testament import Testament

class ReSign(tests.TestCaseInTempDir):

    def monkey_patch_gpg(self):
        if False:
            for i in range(10):
                print('nop')
        'Monkey patch the gpg signing strategy to be a loopback.\n\n        This also registers the cleanup, so that we will revert to\n        the original gpg strategy when done.\n        '
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def setup_tree(self):
        if False:
            i = 10
            return i + 15
        wt = ControlDir.create_standalone_workingtree('.')
        wt.commit('base A', allow_pointless=True, rev_id='A')
        wt.commit('base B', allow_pointless=True, rev_id='B')
        wt.commit('base C', allow_pointless=True, rev_id='C')
        return wt

    def assertEqualSignature(self, repo, revision_id):
        if False:
            return 10
        'Assert a signature is stored correctly in repository.'
        self.assertEqual('-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + Testament.from_revision(repo, revision_id).as_short_text() + '-----END PSEUDO-SIGNED CONTENT-----\n', repo.get_signature_text(revision_id))

    def test_resign(self):
        if False:
            while True:
                i = 10
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign -r revid:A')
        self.assertEqualSignature(repo, 'A')
        self.run_bzr('re-sign B')
        self.assertEqualSignature(repo, 'B')

    def test_resign_range(self):
        if False:
            i = 10
            return i + 15
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign -r 1..')
        self.assertEqualSignature(repo, 'A')
        self.assertEqualSignature(repo, 'B')
        self.assertEqualSignature(repo, 'C')

    def test_resign_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign A B C')
        self.assertEqualSignature(repo, 'A')
        self.assertEqualSignature(repo, 'B')
        self.assertEqualSignature(repo, 'C')

    def test_resign_directory(self):
        if False:
            return 10
        'Test --directory option'
        wt = ControlDir.create_standalone_workingtree('a')
        wt.commit('base A', allow_pointless=True, rev_id='A')
        wt.commit('base B', allow_pointless=True, rev_id='B')
        wt.commit('base C', allow_pointless=True, rev_id='C')
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign --directory=a -r revid:A')
        self.assertEqualSignature(repo, 'A')
        self.run_bzr('re-sign -d a B')
        self.assertEqualSignature(repo, 'B')