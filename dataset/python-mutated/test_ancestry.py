import os
from bzrlib.tests import TestCaseWithTransport

class TestAncestry(TestCaseWithTransport):

    def _build_branches(self):
        if False:
            for i in range(10):
                print('nop')
        a_wt = self.make_branch_and_tree('A')
        self.build_tree_contents([('A/foo', '1111\n')])
        a_wt.add('foo')
        a_wt.commit('added foo', rev_id='A1')
        b_wt = a_wt.bzrdir.sprout('B').open_workingtree()
        self.build_tree_contents([('B/foo', '1111\n22\n')])
        b_wt.commit('modified B/foo', rev_id='B1')
        self.build_tree_contents([('A/foo', '000\n1111\n')])
        a_wt.commit('modified A/foo', rev_id='A2')
        a_wt.merge_from_branch(b_wt.branch, b_wt.last_revision(), b_wt.branch.get_rev_id(1))
        a_wt.commit('merged B into A', rev_id='A3')
        return (a_wt, b_wt)

    def _check_ancestry(self, location='', result=None):
        if False:
            print('Hello World!')
        out = self.run_bzr(['ancestry', location])[0]
        if result is not None:
            self.assertEqualDiff(result, out)
        else:
            result = 'A1\nB1\nA2\nA3\n'
            if result != out:
                result = 'A1\nA2\nB1\nA3\n'
            self.assertEqualDiff(result, out)

    def test_ancestry(self):
        if False:
            print('Hello World!')
        "Tests 'ancestry' command"
        self._build_branches()
        os.chdir('A')
        self._check_ancestry()

    def test_ancestry_with_location(self):
        if False:
            return 10
        "Tests 'ancestry' command with a specified location."
        self._build_branches()
        self._check_ancestry('A')

    def test_ancestry_with_repo_branch(self):
        if False:
            print('Hello World!')
        "Tests 'ancestry' command with a location that is a\n        repository branch."
        a_tree = self._build_branches()[0]
        self.make_repository('repo', shared=True)
        a_tree.bzrdir.sprout('repo/A')
        self._check_ancestry('repo/A')

    def test_ancestry_with_checkout(self):
        if False:
            return 10
        "Tests 'ancestry' command with a location that is a\n        checkout of a repository branch."
        a_tree = self._build_branches()[0]
        self.make_repository('repo', shared=True)
        repo_branch = a_tree.bzrdir.sprout('repo/A').open_branch()
        repo_branch.create_checkout('A-checkout')
        self._check_ancestry('A-checkout')

    def test_ancestry_with_lightweight_checkout(self):
        if False:
            print('Hello World!')
        "Tests 'ancestry' command with a location that is a\n        lightweight checkout of a repository branch."
        a_tree = self._build_branches()[0]
        self.make_repository('repo', shared=True)
        repo_branch = a_tree.bzrdir.sprout('repo/A').open_branch()
        repo_branch.create_checkout('A-checkout', lightweight=True)
        self._check_ancestry('A-checkout')

    def test_ancestry_with_truncated_checkout(self):
        if False:
            return 10
        "Tests 'ancestry' command with a location that is a\n        checkout of a repository branch with a shortened revision history."
        a_tree = self._build_branches()[0]
        self.make_repository('repo', shared=True)
        repo_branch = a_tree.bzrdir.sprout('repo/A').open_branch()
        repo_branch.create_checkout('A-checkout', revision_id=repo_branch.get_rev_id(2))
        self._check_ancestry('A-checkout', 'A1\nA2\n')

    def test_ancestry_with_truncated_lightweight_checkout(self):
        if False:
            while True:
                i = 10
        "Tests 'ancestry' command with a location that is a lightweight\n        checkout of a repository branch with a shortened revision history."
        a_tree = self._build_branches()[0]
        self.make_repository('repo', shared=True)
        repo_branch = a_tree.bzrdir.sprout('repo/A').open_branch()
        repo_branch.create_checkout('A-checkout', revision_id=repo_branch.get_rev_id(2), lightweight=True)
        self._check_ancestry('A-checkout', 'A1\nA2\n')