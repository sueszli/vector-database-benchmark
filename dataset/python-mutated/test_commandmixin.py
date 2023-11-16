from twisted.internet import defer
from buildbot.process import results
from buildbot.process.buildstep import BuildStep
from buildbot.process.buildstep import CommandMixin
from buildbot.test.util.integration import RunMasterBase

class CommandMixinMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            return 10
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(TestCommandMixinStep())
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_commandmixin(self):
        if False:
            print('Hello World!')
        yield self.setup_config()
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['results'], results.SUCCESS)

class CommandMixinMasterPB(CommandMixinMaster):
    proto = 'pb'

class CommandMixinMasterMsgPack(CommandMixinMaster):
    proto = 'msgpack'

class TestCommandMixinStep(BuildStep, CommandMixin):

    @defer.inlineCallbacks
    def run(self):
        if False:
            for i in range(10):
                print('nop')
        contents = (yield self.runGlob('*'))
        if contents != []:
            return results.FAILURE
        hasPath = (yield self.pathExists('composite_mixin_test'))
        if hasPath:
            return results.FAILURE
        yield self.runMkdir('composite_mixin_test')
        hasPath = (yield self.pathExists('composite_mixin_test'))
        if not hasPath:
            return results.FAILURE
        contents = (yield self.runGlob('*'))
        if not contents[0].endswith('composite_mixin_test'):
            return results.FAILURE
        yield self.runRmdir('composite_mixin_test')
        hasPath = (yield self.pathExists('composite_mixin_test'))
        if hasPath:
            return results.FAILURE
        return results.SUCCESS