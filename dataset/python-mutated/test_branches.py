"""Black-box tests for bzr branches."""
from bzrlib.bzrdir import BzrDir
from bzrlib.tests import TestCaseWithTransport

class TestBranches(TestCaseWithTransport):

    def test_no_colocated_support(self):
        if False:
            print('Hello World!')
        self.run_bzr('init a')
        (out, err) = self.run_bzr('branches a')
        self.assertEqual(out, '* (default)\n')

    def test_no_branch(self):
        if False:
            i = 10
            return i + 15
        self.run_bzr('init-repo a')
        (out, err) = self.run_bzr('branches a')
        self.assertEqual(out, '')

    def test_default_current_dir(self):
        if False:
            i = 10
            return i + 15
        self.run_bzr('init-repo a')
        (out, err) = self.run_bzr('branches', working_dir='a')
        self.assertEqual(out, '')

    def test_recursive_current(self):
        if False:
            while True:
                i = 10
        self.run_bzr('init .')
        self.assertEqual('.\n', self.run_bzr('branches --recursive')[0])

    def test_recursive(self):
        if False:
            print('Hello World!')
        self.run_bzr('init source')
        self.run_bzr('init source/subsource')
        self.run_bzr('checkout --lightweight source checkout')
        self.run_bzr('init checkout/subcheckout')
        self.run_bzr('init checkout/.bzr/subcheckout')
        out = self.run_bzr('branches --recursive')[0]
        lines = out.split('\n')
        self.assertIs(True, 'source' in lines, lines)
        self.assertIs(True, 'source/subsource' in lines, lines)
        self.assertIs(True, 'checkout/subcheckout' in lines, lines)
        self.assertIs(True, 'checkout' not in lines, lines)

    def test_indicates_non_branch(self):
        if False:
            while True:
                i = 10
        t = self.make_branch_and_tree('a', format='development-colo')
        t.bzrdir.create_branch(name='another')
        t.bzrdir.create_branch(name='colocated')
        (out, err) = self.run_bzr('branches a')
        self.assertEqual(out, '* (default)\n  another\n  colocated\n')

    def test_indicates_branch(self):
        if False:
            while True:
                i = 10
        t = self.make_repository('a', format='development-colo')
        t.bzrdir.create_branch(name='another')
        branch = t.bzrdir.create_branch(name='colocated')
        t.bzrdir.set_branch_reference(target_branch=branch)
        (out, err) = self.run_bzr('branches a')
        self.assertEqual(out, '  another\n* colocated\n')

    def test_shared_repos(self):
        if False:
            return 10
        self.make_repository('a', shared=True)
        BzrDir.create_branch_convenience('a/branch1')
        b = BzrDir.create_branch_convenience('a/branch2')
        b.create_checkout(lightweight=True, to_location='b')
        (out, err) = self.run_bzr('branches b')
        self.assertEqual(out, '  branch1\n* branch2\n')

    def test_standalone_branch(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_branch('a')
        (out, err) = self.run_bzr('branches a')
        self.assertEqual(out, '* (default)\n')