from twisted.trial import unittest
from buildbot import config
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.steps import maxq
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class TestShellCommandExecution(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tear_down_test_build_step()

    def test_testdir_required(self):
        if False:
            return 10
        with self.assertRaises(config.ConfigErrors):
            maxq.MaxQ()

    def test_success(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(maxq.MaxQ(testdir='x'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['run_maxq.py', 'x']).stdout('no failures\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='success')
        return self.run_step()

    def test_nonzero_rc_no_failures(self):
        if False:
            return 10
        self.setup_step(maxq.MaxQ(testdir='x'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['run_maxq.py', 'x']).stdout('no failures\n').exit(2))
        self.expect_outcome(result=FAILURE, state_string='1 maxq failures')
        return self.run_step()

    def test_failures(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(maxq.MaxQ(testdir='x'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['run_maxq.py', 'x']).stdout('\nTEST FAILURE: foo\n' * 10).exit(2))
        self.expect_outcome(result=FAILURE, state_string='10 maxq failures')
        return self.run_step()