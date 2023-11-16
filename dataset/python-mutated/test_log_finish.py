from twisted.internet import defer
from buildbot.plugins import steps
from buildbot.process.results import EXCEPTION
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase

class TestLog(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, step):
        if False:
            return 10
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(step)
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_shellcommand(self):
        if False:
            i = 10
            return i + 15
        testcase = self

        class MyStep(steps.ShellCommand):

            def _newLog(self, name, type, logid, logEncoding):
                if False:
                    for i in range(10):
                        print('nop')
                r = super()._newLog(name, type, logid, logEncoding)
                testcase.curr_log = r
                return r
        step = MyStep(command='echo hello')
        yield self.setup_config(step)
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['results'], SUCCESS)
        self.assertTrue(self.curr_log.finished)

    @defer.inlineCallbacks
    def test_mastershellcommand(self):
        if False:
            i = 10
            return i + 15
        testcase = self

        class MyStep(steps.MasterShellCommand):

            def _newLog(self, name, type, logid, logEncoding):
                if False:
                    print('Hello World!')
                r = super()._newLog(name, type, logid, logEncoding)
                testcase.curr_log = r
                return r
        step = MyStep(command='echo hello')
        yield self.setup_config(step)
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['results'], SUCCESS)
        self.assertTrue(self.curr_log.finished)

    @defer.inlineCallbacks
    def test_mastershellcommand_issue(self):
        if False:
            return 10
        testcase = self

        class MyStep(steps.MasterShellCommand):

            def _newLog(self, name, type, logid, logEncoding):
                if False:
                    for i in range(10):
                        print('nop')
                r = super()._newLog(name, type, logid, logEncoding)
                testcase.curr_log = r
                testcase.patch(r, 'finish', lambda : defer.fail(RuntimeError('Could not finish')))
                return r
        step = MyStep(command='echo hello')
        yield self.setup_config(step)
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertFalse(self.curr_log.finished)
        self.assertEqual(build['results'], EXCEPTION)
        errors = self.flushLoggedErrors()
        self.assertEqual(len(errors), 1)
        error = errors[0]
        self.assertEqual(error.getErrorMessage(), 'Could not finish')