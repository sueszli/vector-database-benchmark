from twisted.trial import unittest
from buildbot.process.results import SUCCESS
from buildbot.steps.package.rpm import rpmlint
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class TestRpmLint(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tear_down_test_build_step()

    def test_success(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(rpmlint.RpmLint())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['rpmlint', '-i', '.']).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='Finished checking RPM/SPEC issues')
        return self.run_step()

    def test_fileloc_success(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(rpmlint.RpmLint(fileloc='RESULT'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['rpmlint', '-i', 'RESULT']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_config_success(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(rpmlint.RpmLint(config='foo.cfg'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['rpmlint', '-i', '-f', 'foo.cfg', '.']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()