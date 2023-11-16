"""Black-box tests for bzr rmbranch."""
from bzrlib import controldir
from bzrlib.tests import TestCaseWithTransport
from bzrlib.tests.matchers import ContainsNoVfsCalls

class TestRemoveBranch(TestCaseWithTransport):

    def example_tree(self, path='.', format=None):
        if False:
            return 10
        tree = self.make_branch_and_tree(path, format=format)
        self.build_tree_contents([(path + '/hello', 'foo')])
        tree.add('hello')
        tree.commit(message='setup')
        self.build_tree_contents([(path + '/goodbye', 'baz')])
        tree.add('goodbye')
        tree.commit(message='setup')
        return tree

    def test_remove_local(self):
        if False:
            print('Hello World!')
        tree = self.example_tree('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
        self.run_bzr('rmbranch --force a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())
        self.assertPathExists('a/hello')
        self.assertPathExists('a/goodbye')

    def test_no_branch(self):
        if False:
            return 10
        self.make_repository('a')
        self.run_bzr_error(['Not a branch'], 'rmbranch a')

    def test_no_tree(self):
        if False:
            print('Hello World!')
        tree = self.example_tree('a')
        tree.bzrdir.destroy_workingtree()
        self.run_bzr('rmbranch', working_dir='a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())

    def test_no_arg(self):
        if False:
            for i in range(10):
                print('nop')
        self.example_tree('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
        self.run_bzr('rmbranch --force', working_dir='a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())

    def test_remove_colo(self):
        if False:
            print('Hello World!')
        tree = self.example_tree('a')
        tree.bzrdir.create_branch(name='otherbranch')
        self.assertTrue(tree.bzrdir.has_branch('otherbranch'))
        self.run_bzr('rmbranch %s,branch=otherbranch' % tree.bzrdir.user_url)
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch('otherbranch'))
        self.assertTrue(dir.has_branch())

    def test_remove_colo_directory(self):
        if False:
            while True:
                i = 10
        tree = self.example_tree('a')
        tree.bzrdir.create_branch(name='otherbranch')
        self.assertTrue(tree.bzrdir.has_branch('otherbranch'))
        self.run_bzr('rmbranch otherbranch -d %s' % tree.bzrdir.user_url)
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch('otherbranch'))
        self.assertTrue(dir.has_branch())

    def test_remove_active_colo_branch(self):
        if False:
            i = 10
            return i + 15
        dir = self.make_repository('a').bzrdir
        branch = dir.create_branch('otherbranch')
        branch.create_checkout('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch otherbranch -d %s' % branch.bzrdir.user_url)
        self.assertTrue(dir.has_branch('otherbranch'))
        self.run_bzr('rmbranch --force otherbranch -d %s' % branch.bzrdir.user_url)
        self.assertFalse(dir.has_branch('otherbranch'))

class TestSmartServerRemoveBranch(TestCaseWithTransport):

    def test_simple_remove_branch(self):
        if False:
            print('Hello World!')
        self.setup_smart_server_with_call_log()
        self.make_branch('branch')
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['rmbranch', self.get_url('branch')])
        self.assertLength(5, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)