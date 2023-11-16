from twisted.trial import unittest
from buildbot import config
from buildbot.process.properties import Interpolate
from buildbot.process.results import SUCCESS
from buildbot.steps.package.rpm import mock
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class TestMock(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            return 10
        return self.tear_down_test_build_step()

    def test_no_root(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(config.ConfigErrors):
            mock.Mock()

    def test_class_attrs(self):
        if False:
            print('Hello World!')
        step = self.setup_step(mock.Mock(root='TESTROOT'))
        self.assertEqual(step.command, ['mock', '--root', 'TESTROOT'])

    def test_success(self):
        if False:
            print('Hello World!')
        self.setup_step(mock.Mock(root='TESTROOT'))
        self.expect_commands(ExpectRmdir(dir=['build/build.log', 'build/root.log', 'build/state.log'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['mock', '--root', 'TESTROOT'], logfiles={'build.log': 'build.log', 'root.log': 'root.log', 'state.log': 'state.log'}).exit(0))
        self.expect_outcome(result=SUCCESS, state_string="'mock --root ...'")
        return self.run_step()

    def test_resultdir_success(self):
        if False:
            return 10
        self.setup_step(mock.Mock(root='TESTROOT', resultdir='RESULT'))
        self.expect_commands(ExpectRmdir(dir=['build/RESULT/build.log', 'build/RESULT/root.log', 'build/RESULT/state.log'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['mock', '--root', 'TESTROOT', '--resultdir', 'RESULT'], logfiles={'build.log': 'RESULT/build.log', 'root.log': 'RESULT/root.log', 'state.log': 'RESULT/state.log'}).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_resultdir_renderable(self):
        if False:
            i = 10
            return i + 15
        resultdir_text = 'RESULT'
        self.setup_step(mock.Mock(root='TESTROOT', resultdir=Interpolate('%(kw:resultdir)s', resultdir=resultdir_text)))
        self.expect_commands(ExpectRmdir(dir=['build/RESULT/build.log', 'build/RESULT/root.log', 'build/RESULT/state.log'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['mock', '--root', 'TESTROOT', '--resultdir', 'RESULT'], logfiles={'build.log': 'RESULT/build.log', 'root.log': 'RESULT/root.log', 'state.log': 'RESULT/state.log'}).exit(0))
        self.expect_outcome(result=SUCCESS, state_string="'mock --root ...'")
        return self.run_step()

class TestMockBuildSRPM(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            return 10
        return self.tear_down_test_build_step()

    def test_no_spec(self):
        if False:
            print('Hello World!')
        with self.assertRaises(config.ConfigErrors):
            mock.MockBuildSRPM(root='TESTROOT')

    def test_success(self):
        if False:
            while True:
                i = 10
        self.setup_step(mock.MockBuildSRPM(root='TESTROOT', spec='foo.spec'))
        self.expect_commands(ExpectRmdir(dir=['build/build.log', 'build/root.log', 'build/state.log'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['mock', '--root', 'TESTROOT', '--buildsrpm', '--spec', 'foo.spec', '--sources', '.'], logfiles={'build.log': 'build.log', 'root.log': 'root.log', 'state.log': 'state.log'}).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='mock buildsrpm')
        return self.run_step()

class TestMockRebuild(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def test_no_srpm(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(config.ConfigErrors):
            mock.MockRebuild(root='TESTROOT')

    def test_success(self):
        if False:
            while True:
                i = 10
        self.setup_step(mock.MockRebuild(root='TESTROOT', srpm='foo.src.rpm'))
        self.expect_commands(ExpectRmdir(dir=['build/build.log', 'build/root.log', 'build/state.log'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['mock', '--root', 'TESTROOT', '--rebuild', 'foo.src.rpm'], logfiles={'build.log': 'build.log', 'root.log': 'root.log', 'state.log': 'state.log'}).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='mock rebuild srpm')
        return self.run_step()