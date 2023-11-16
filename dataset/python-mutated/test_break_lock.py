"""Tests for lock-breaking user interface"""
from bzrlib import branch, config, controldir, errors, osutils, tests
from bzrlib.tests.matchers import ContainsNoVfsCalls
from bzrlib.tests.script import run_script

class TestBreakLock(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestBreakLock, self).setUp()
        self.build_tree(['master-repo/', 'master-repo/master-branch/', 'repo/', 'repo/branch/', 'checkout/'])
        controldir.ControlDir.create('master-repo').create_repository()
        self.master_branch = controldir.ControlDir.create_branch_convenience('master-repo/master-branch')
        controldir.ControlDir.create('repo').create_repository()
        local_branch = controldir.ControlDir.create_branch_convenience('repo/branch')
        local_branch.bind(self.master_branch)
        checkoutdir = controldir.ControlDir.create('checkout')
        checkoutdir.set_branch_reference(local_branch)
        self.wt = checkoutdir.create_workingtree()

    def test_break_lock_help(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr('break-lock --help')
        self.assertEqual('', err)

    def test_break_lock_no_interaction(self):
        if False:
            while True:
                i = 10
        "With --force, the user isn't asked for confirmation"
        self.master_branch.lock_write()
        run_script(self, '\n        $ bzr break-lock --force master-repo/master-branch\n        Broke lock ...master-branch/.bzr/...\n        ')
        self.assertRaises(errors.LockBroken, self.master_branch.unlock)

    def test_break_lock_everything_locked(self):
        if False:
            return 10
        self.wt.branch.lock_write()
        self.master_branch.lock_write()
        self.run_bzr('break-lock checkout', stdin='y\ny\ny\ny\n')
        br = branch.Branch.open('checkout')
        br.lock_write()
        br.unlock()
        mb = br.get_master_branch()
        mb.lock_write()
        mb.unlock()
        self.assertRaises(errors.LockBroken, self.wt.unlock)
        self.assertRaises(errors.LockBroken, self.master_branch.unlock)

class TestConfigBreakLock(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestConfigBreakLock, self).setUp()
        self.config_file_name = './my.conf'
        self.build_tree_contents([(self.config_file_name, '[DEFAULT]\none=1\n')])
        self.config = config.LockableConfig(file_name=self.config_file_name)
        self.config.lock_write()

    def test_create_pending_lock(self):
        if False:
            i = 10
            return i + 15
        self.addCleanup(self.config.unlock)
        self.assertTrue(self.config._lock.is_held)

    def test_break_lock(self):
        if False:
            print('Hello World!')
        self.run_bzr('break-lock --config %s' % osutils.dirname(self.config_file_name), stdin='y\n')
        self.assertRaises(errors.LockBroken, self.config.unlock)

class TestSmartServerBreakLock(tests.TestCaseWithTransport):

    def test_simple_branch_break_lock(self):
        if False:
            i = 10
            return i + 15
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        t.branch.lock_write()
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['break-lock', '--force', self.get_url('branch')])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(1, self.hpss_connections)
        self.assertLength(5, self.hpss_calls)