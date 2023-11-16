from collections import OrderedDict
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot import config
from buildbot.process.properties import Interpolate
from buildbot.process.results import SUCCESS
from buildbot.steps.package.rpm import rpmbuild
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class RpmBuild(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tear_down_test_build_step()

    def test_no_specfile(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(config.ConfigErrors):
            rpmbuild.RpmBuild()

    def test_success(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(rpmbuild.RpmBuild(specfile='foo.spec', dist='.el5'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='rpmbuild --define "_topdir `pwd`" --define "_builddir `pwd`" --define "_rpmdir `pwd`" --define "_sourcedir `pwd`" --define "_specdir `pwd`" --define "_srcrpmdir `pwd`" --define "dist .el5" -ba foo.spec').stdout('lalala').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='RPMBUILD')
        return self.run_step()

    @mock.patch('builtins.open', mock.mock_open())
    def test_autoRelease(self):
        if False:
            while True:
                i = 10
        self.setup_step(rpmbuild.RpmBuild(specfile='foo.spec', autoRelease=True))
        self.expect_commands(ExpectShell(workdir='wkdir', command='rpmbuild --define "_topdir `pwd`" --define "_builddir `pwd`" --define "_rpmdir `pwd`" --define "_sourcedir `pwd`" --define "_specdir `pwd`" --define "_srcrpmdir `pwd`" --define "_release 0" --define "dist .el6" -ba foo.spec').stdout('Your code has been rated at 10/10').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='RPMBUILD')
        return self.run_step()

    def test_define(self):
        if False:
            for i in range(10):
                print('nop')
        defines = [('a', '1'), ('b', '2')]
        self.setup_step(rpmbuild.RpmBuild(specfile='foo.spec', define=OrderedDict(defines)))
        self.expect_commands(ExpectShell(workdir='wkdir', command='rpmbuild --define "_topdir `pwd`" --define "_builddir `pwd`" --define "_rpmdir `pwd`" --define "_sourcedir `pwd`" --define "_specdir `pwd`" --define "_srcrpmdir `pwd`" --define "a 1" --define "b 2" --define "dist .el6" -ba foo.spec').stdout('Your code has been rated at 10/10').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='RPMBUILD')
        return self.run_step()

    def test_define_none(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(rpmbuild.RpmBuild(specfile='foo.spec', define=None))
        self.expect_commands(ExpectShell(workdir='wkdir', command='rpmbuild --define "_topdir `pwd`" --define "_builddir `pwd`" --define "_rpmdir `pwd`" --define "_sourcedir `pwd`" --define "_specdir `pwd`" --define "_srcrpmdir `pwd`" --define "dist .el6" -ba foo.spec').stdout('Your code has been rated at 10/10').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='RPMBUILD')
        return self.run_step()

    @defer.inlineCallbacks
    def test_renderable_dist(self):
        if False:
            print('Hello World!')
        self.setup_step(rpmbuild.RpmBuild(specfile='foo.spec', dist=Interpolate('%(prop:renderable_dist)s')))
        self.properties.setProperty('renderable_dist', '.el7', 'test')
        self.expect_commands(ExpectShell(workdir='wkdir', command='rpmbuild --define "_topdir `pwd`" --define "_builddir `pwd`" --define "_rpmdir `pwd`" --define "_sourcedir `pwd`" --define "_specdir `pwd`" --define "_srcrpmdir `pwd`" --define "dist .el7" -ba foo.spec').stdout('lalala').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='RPMBUILD')
        yield self.run_step()