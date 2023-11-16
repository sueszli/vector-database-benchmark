from twisted.internet import defer
from buildbot.plugins import schedulers
from buildbot.test.util.integration import RunMasterBase

class ShellMaster(RunMasterBase):

    def create_config(self):
        if False:
            return 10
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched1', builderNames=['testy1']), schedulers.ForceScheduler(name='sched2', builderNames=['testy2'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='echo hello'))
        c['builders'] = [BuilderConfig(name=name, workernames=['local1'], factory=f) for name in ['testy1', 'testy2']]
        return c

    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            return 10
        cfg = self.create_config()
        yield self.setup_master(cfg)
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        cfg['schedulers'] = [schedulers.AnyBranchScheduler(name='sched1', builderNames=['testy2']), schedulers.ForceScheduler(name='sched2', builderNames=['testy1'])]
        yield self.master.reconfig()
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        builder = (yield self.master.data.get(('builders', build['builderid'])))
        self.assertEqual(builder['name'], 'testy2')