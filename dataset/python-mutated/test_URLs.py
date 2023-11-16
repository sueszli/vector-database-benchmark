from twisted.internet import defer
from twisted.python import runtime
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase

class UrlForBuildMaster(RunMasterBase):
    proto = 'null'

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.plugins import util
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command=['echo', util.URLForBuild]))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_url(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['results'], SUCCESS)
        if runtime.platformType == 'win32':
            command = 'echo http://localhost:8080/#/builders/1/builds/1'
        else:
            command = "echo 'http://localhost:8080/#/builders/1/builds/1'"
        self.assertIn(command, build['steps'][1]['logs'][0]['contents']['content'])