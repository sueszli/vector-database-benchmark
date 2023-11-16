from bzrlib import branch, tests

class TestRevisionHistory(tests.TestCaseWithTransport):

    def _build_branch(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('test')
        with open('test/foo', 'wb') as f:
            f.write('1111\n')
        tree.add('foo')
        tree.commit('added foo', rev_id='revision_1')
        with open('test/foo', 'wb') as f:
            f.write('2222\n')
        tree.commit('updated foo', rev_id='revision_2')
        with open('test/foo', 'wb') as f:
            f.write('3333\n')
        tree.commit('updated foo again', rev_id='revision_3')
        return tree

    def _check_revision_history(self, location='', working_dir=None):
        if False:
            i = 10
            return i + 15
        rh = self.run_bzr(['revision-history', location], working_dir=working_dir)[0]
        self.assertEqual(rh, 'revision_1\nrevision_2\nrevision_3\n')

    def test_revision_history(self):
        if False:
            while True:
                i = 10
        'No location'
        self._build_branch()
        self._check_revision_history(working_dir='test')

    def test_revision_history_with_location(self):
        if False:
            print('Hello World!')
        'With a specified location.'
        self._build_branch()
        self._check_revision_history('test')

    def test_revision_history_with_repo_branch(self):
        if False:
            i = 10
            return i + 15
        'With a repository branch location.'
        self._build_branch()
        self.run_bzr('init-repo repo')
        self.run_bzr('branch test repo/test')
        self._check_revision_history('repo/test')

    def test_revision_history_with_checkout(self):
        if False:
            for i in range(10):
                print('nop')
        'With a repository branch checkout location.'
        self._build_branch()
        self.run_bzr('init-repo repo')
        self.run_bzr('branch test repo/test')
        self.run_bzr('checkout repo/test test-checkout')
        self._check_revision_history('test-checkout')

    def test_revision_history_with_lightweight_checkout(self):
        if False:
            for i in range(10):
                print('nop')
        'With a repository branch lightweight checkout location.'
        self._build_branch()
        self.run_bzr('init-repo repo')
        self.run_bzr('branch test repo/test')
        self.run_bzr('checkout --lightweight repo/test test-checkout')
        self._check_revision_history('test-checkout')