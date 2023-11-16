import sys
import textwrap
from twisted.internet import defer
from twisted.internet import reactor
from buildbot.config import BuilderConfig
from buildbot.plugins import schedulers
from buildbot.plugins import steps
from buildbot.process.factory import BuildFactory
from buildbot.process.results import CANCELLED
from buildbot.test.util.integration import RunMasterBase

class TriggeringMaster(RunMasterBase):
    timeout = 120
    change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}

    @defer.inlineCallbacks
    def setup_trigger_config(self, triggeredFactory, nextBuild=None):
        if False:
            i = 10
            return i + 15
        c = {}
        c['schedulers'] = [schedulers.Triggerable(name='trigsched', builderNames=['triggered']), schedulers.AnyBranchScheduler(name='sched', builderNames=['main'])]
        f = BuildFactory()
        f.addStep(steps.Trigger(schedulerNames=['trigsched'], waitForFinish=True, updateSourceStamp=True))
        f.addStep(steps.ShellCommand(command='echo world'))
        mainBuilder = BuilderConfig(name='main', workernames=['local1'], factory=f)
        triggeredBuilderKwargs = {'name': 'triggered', 'workernames': ['local1'], 'factory': triggeredFactory}
        if nextBuild is not None:
            triggeredBuilderKwargs['nextBuild'] = nextBuild
        triggeredBuilder = BuilderConfig(**triggeredBuilderKwargs)
        c['builders'] = [mainBuilder, triggeredBuilder]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def setup_config_trigger_runs_forever(self):
        if False:
            print('Hello World!')
        f2 = BuildFactory()
        if sys.platform == 'win32':
            cmd = 'ping -t 127.0.0.1'.split()
        else:
            cmd = textwrap.dedent('                while :\n                do\n                  echo "sleeping";\n                  sleep 1;\n                done\n                ')
        f2.addStep(steps.ShellCommand(command=cmd))
        yield self.setup_trigger_config(f2)

    @defer.inlineCallbacks
    def setup_config_triggered_build_not_created(self):
        if False:
            while True:
                i = 10
        f2 = BuildFactory()
        f2.addStep(steps.ShellCommand(command="echo 'hello'"))

        def nextBuild(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return defer.succeed(None)
        yield self.setup_trigger_config(f2, nextBuild=nextBuild)

    def assertBuildIsCancelled(self, b):
        if False:
            while True:
                i = 10
        self.assertTrue(b['complete'])
        self.assertEqual(b['results'], CANCELLED, repr(b))

    @defer.inlineCallbacks
    def runTest(self, newBuildCallback, flushErrors=False):
        if False:
            while True:
                i = 10
        newConsumer = (yield self.master.mq.startConsuming(newBuildCallback, ('builds', None, 'new')))
        build = (yield self.doForceBuild(wantSteps=True, useChange=self.change, wantLogs=True))
        self.assertBuildIsCancelled(build)
        newConsumer.stopConsuming()
        builds = (yield self.master.data.get(('builds',)))
        for b in builds:
            self.assertBuildIsCancelled(b)
        if flushErrors:
            self.flushLoggedErrors()

    @defer.inlineCallbacks
    def testTriggerRunsForever(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.setup_config_trigger_runs_forever()
        self.higherBuild = None

        def newCallback(_, data):
            if False:
                return 10
            if self.higherBuild is None:
                self.higherBuild = data['buildid']
            else:
                self.master.data.control('stop', {}, ('builds', self.higherBuild))
                self.higherBuild = None
        yield self.runTest(newCallback, flushErrors=True)

    @defer.inlineCallbacks
    def testTriggerRunsForeverAfterCmdStarted(self):
        if False:
            print('Hello World!')
        yield self.setup_config_trigger_runs_forever()
        self.higherBuild = None

        def newCallback(_, data):
            if False:
                while True:
                    i = 10
            if self.higherBuild is None:
                self.higherBuild = data['buildid']
            else:

                def f():
                    if False:
                        for i in range(10):
                            print('nop')
                    self.master.data.control('stop', {}, ('builds', self.higherBuild))
                    self.higherBuild = None
                reactor.callLater(5.0, f)
        yield self.runTest(newCallback, flushErrors=True)

    @defer.inlineCallbacks
    def testTriggeredBuildIsNotCreated(self):
        if False:
            i = 10
            return i + 15
        yield self.setup_config_triggered_build_not_created()

        def newCallback(_, data):
            if False:
                return 10
            self.master.data.control('stop', {}, ('builds', data['buildid']))
        yield self.runTest(newCallback)