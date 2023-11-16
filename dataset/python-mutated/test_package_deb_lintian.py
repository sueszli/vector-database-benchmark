from twisted.trial import unittest
from buildbot import config
from buildbot.process.results import SUCCESS
from buildbot.steps.package.deb import lintian
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class TestDebLintian(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

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

    def test_no_fileloc(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(config.ConfigErrors):
            lintian.DebLintian()

    def test_success(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(lintian.DebLintian('foo_0.23_i386.changes'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['lintian', '-v', 'foo_0.23_i386.changes']).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='Lintian')
        return self.run_step()

    def test_success_suppressTags(self):
        if False:
            while True:
                i = 10
        self.setup_step(lintian.DebLintian('foo_0.23_i386.changes', suppressTags=['bad-distribution-in-changes-file']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['lintian', '-v', 'foo_0.23_i386.changes', '--suppress-tags', 'bad-distribution-in-changes-file']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()