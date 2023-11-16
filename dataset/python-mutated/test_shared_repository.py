"""Black-box tests for repositories with shared branches"""
import os
from bzrlib.bzrdir import BzrDirMetaFormat1
from bzrlib.controldir import ControlDir
import bzrlib.errors as errors
from bzrlib.tests import TestCaseInTempDir
from bzrlib.tests.matchers import ContainsNoVfsCalls

class TestSharedRepo(TestCaseInTempDir):

    def test_make_repository(self):
        if False:
            return 10
        (out, err) = self.run_bzr('init-repository a')
        self.assertEqual(out, 'Shared repository with trees (format: 2a)\nLocation:\n  shared repository: a\n')
        self.assertEqual(err, '')
        dir = ControlDir.open('a')
        self.assertIs(dir.open_repository().is_shared(), True)
        self.assertRaises(errors.NotBranchError, dir.open_branch)
        self.assertRaises(errors.NoWorkingTree, dir.open_workingtree)

    def test_make_repository_quiet(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr('init-repository a -q')
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        dir = ControlDir.open('a')
        self.assertIs(dir.open_repository().is_shared(), True)
        self.assertRaises(errors.NotBranchError, dir.open_branch)
        self.assertRaises(errors.NoWorkingTree, dir.open_workingtree)

    def test_init_repo_existing_dir(self):
        if False:
            while True:
                i = 10
        'Make repo in existing directory.\n\n        (Malone #38331)\n        '
        (out, err) = self.run_bzr('init-repository .')
        dir = ControlDir.open('.')
        self.assertTrue(dir.open_repository())

    def test_init(self):
        if False:
            while True:
                i = 10
        self.run_bzr('init-repo a')
        self.run_bzr('init --format=default a/b')
        dir = ControlDir.open('a')
        self.assertIs(dir.open_repository().is_shared(), True)
        self.assertRaises(errors.NotBranchError, dir.open_branch)
        self.assertRaises(errors.NoWorkingTree, dir.open_workingtree)
        bdir = ControlDir.open('a/b')
        bdir.open_branch()
        self.assertRaises(errors.NoRepositoryPresent, bdir.open_repository)
        wt = bdir.open_workingtree()

    def test_branch(self):
        if False:
            i = 10
            return i + 15
        self.run_bzr('init-repo a')
        self.run_bzr('init --format=default a/b')
        self.run_bzr('branch a/b a/c')
        cdir = ControlDir.open('a/c')
        cdir.open_branch()
        self.assertRaises(errors.NoRepositoryPresent, cdir.open_repository)
        cdir.open_workingtree()

    def test_branch_tree(self):
        if False:
            while True:
                i = 10
        self.run_bzr('init-repo --trees a')
        self.run_bzr('init --format=default b')
        with file('b/hello', 'wt') as f:
            f.write('bar')
        self.run_bzr('add b/hello')
        self.run_bzr('commit -m bar b/hello')
        self.run_bzr('branch b a/c')
        cdir = ControlDir.open('a/c')
        cdir.open_branch()
        self.assertRaises(errors.NoRepositoryPresent, cdir.open_repository)
        self.assertPathExists('a/c/hello')
        cdir.open_workingtree()

    def test_trees_default(self):
        if False:
            while True:
                i = 10
        self.run_bzr('init-repo repo')
        repo = ControlDir.open('repo').open_repository()
        self.assertEqual(True, repo.make_working_trees())

    def test_trees_argument(self):
        if False:
            print('Hello World!')
        self.run_bzr('init-repo --trees trees')
        repo = ControlDir.open('trees').open_repository()
        self.assertEqual(True, repo.make_working_trees())

    def test_no_trees_argument(self):
        if False:
            return 10
        self.run_bzr('init-repo --no-trees notrees')
        repo = ControlDir.open('notrees').open_repository()
        self.assertEqual(False, repo.make_working_trees())

    def test_init_repo_smart_acceptance(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_smart_server_with_call_log()
        self.run_bzr(['init-repo', self.get_url('repo')])
        self.assertLength(11, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_notification_on_branch_from_repository(self):
        if False:
            print('Hello World!')
        (out, err) = self.run_bzr('init-repository -q a')
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        dir = ControlDir.open('a')
        dir.open_repository()
        e = self.assertRaises(errors.NotBranchError, dir.open_branch)
        self.assertContainsRe(str(e), 'location is a repository')

    def test_notification_on_branch_from_nonrepository(self):
        if False:
            i = 10
            return i + 15
        fmt = BzrDirMetaFormat1()
        t = self.get_transport()
        t.mkdir('a')
        dir = fmt.initialize_on_transport(t.clone('a'))
        self.assertRaises(errors.NoRepositoryPresent, dir.open_repository)
        e = self.assertRaises(errors.NotBranchError, dir.open_branch)
        self.assertNotContainsRe(str(e), 'location is a repository')

    def test_init_repo_with_post_repo_init_hook(self):
        if False:
            return 10
        calls = []
        ControlDir.hooks.install_named_hook('post_repo_init', calls.append, None)
        self.assertLength(0, calls)
        self.run_bzr('init-repository a')
        self.assertLength(1, calls)

    def test_init_repo_without_username(self):
        if False:
            while True:
                i = 10
        'Ensure init-repo works if username is not set.\n        '
        self.overrideEnv('EMAIL', None)
        self.overrideEnv('BZR_EMAIL', None)
        (out, err) = self.run_bzr(['init-repo', 'foo'])
        self.assertEqual(err, '')
        self.assertTrue(os.path.exists('foo'))