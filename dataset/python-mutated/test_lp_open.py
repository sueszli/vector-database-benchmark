"""Tests for the launchpad-open command."""
from bzrlib.tests import TestCaseWithTransport

class TestLaunchpadOpen(TestCaseWithTransport):

    def run_open(self, location, retcode=0, working_dir='.'):
        if False:
            i = 10
            return i + 15
        (out, err) = self.run_bzr(['launchpad-open', '--dry-run', location], retcode=retcode, working_dir=working_dir)
        return err.splitlines()

    def test_non_branch(self):
        if False:
            while True:
                i = 10
        self.assertEqual(['bzr: ERROR: . is not registered on Launchpad.'], self.run_open('.', retcode=3))

    def test_no_public_location_no_push_location(self):
        if False:
            print('Hello World!')
        self.make_branch('not-public')
        self.assertEqual(['bzr: ERROR: not-public is not registered on Launchpad.'], self.run_open('not-public', retcode=3))

    def test_non_launchpad_branch(self):
        if False:
            for i in range(10):
                print('nop')
        branch = self.make_branch('non-lp')
        url = 'http://example.com/non-lp'
        branch.set_public_branch(url)
        self.assertEqual(['bzr: ERROR: %s is not registered on Launchpad.' % url], self.run_open('non-lp', retcode=3))

    def test_launchpad_branch_with_public_location(self):
        if False:
            return 10
        branch = self.make_branch('lp')
        branch.set_public_branch('bzr+ssh://bazaar.launchpad.net/~foo/bar/baz')
        self.assertEqual(['Opening https://code.launchpad.net/~foo/bar/baz in web browser'], self.run_open('lp'))

    def test_launchpad_branch_with_public_and_push_location(self):
        if False:
            print('Hello World!')
        branch = self.make_branch('lp')
        branch.lock_write()
        try:
            branch.set_public_branch('bzr+ssh://bazaar.launchpad.net/~foo/bar/public')
            branch.set_push_location('bzr+ssh://bazaar.launchpad.net/~foo/bar/push')
        finally:
            branch.unlock()
        self.assertEqual(['Opening https://code.launchpad.net/~foo/bar/public in web browser'], self.run_open('lp'))

    def test_launchpad_branch_with_no_public_but_with_push(self):
        if False:
            print('Hello World!')
        branch = self.make_branch('lp')
        branch.set_push_location('bzr+ssh://bazaar.launchpad.net/~foo/bar/baz')
        self.assertEqual(['Opening https://code.launchpad.net/~foo/bar/baz in web browser'], self.run_open('lp'))

    def test_launchpad_branch_with_no_public_no_push(self):
        if False:
            return 10
        self.assertEqual(['Opening https://code.launchpad.net/~foo/bar/baz in web browser'], self.run_open('bzr+ssh://bazaar.launchpad.net/~foo/bar/baz'))

    def test_launchpad_branch_subdirectory(self):
        if False:
            return 10
        wt = self.make_branch_and_tree('lp')
        wt.branch.set_push_location('bzr+ssh://bazaar.launchpad.net/~foo/bar/baz')
        self.build_tree(['lp/a/'])
        self.assertEqual(['Opening https://code.launchpad.net/~foo/bar/baz in web browser'], self.run_open('.', working_dir='lp/a'))