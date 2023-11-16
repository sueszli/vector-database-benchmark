from twisted.internet import defer
from buildbot.test.util.integration import RunMasterBase

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy']), schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='echo hello'))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        c['www'] = {'graphql': True}
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            return 10
        yield self.setup_config()
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True, wantProperties=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['properties']['owners'], (['me@foo.com'], 'Build'))