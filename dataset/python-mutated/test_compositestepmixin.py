from twisted.internet import defer
from buildbot.process import results
from buildbot.process.buildstep import BuildStep
from buildbot.steps.worker import CompositeStepMixin
from buildbot.test.util.integration import RunMasterBase

class TestCompositeMixinStep(BuildStep, CompositeStepMixin):

    def __init__(self, is_list_mkdir, is_list_rmdir):
        if False:
            while True:
                i = 10
        super().__init__()
        self.logEnviron = False
        self.is_list_mkdir = is_list_mkdir
        self.is_list_rmdir = is_list_rmdir

    @defer.inlineCallbacks
    def run(self):
        if False:
            return 10
        contents = (yield self.runGlob('*'))
        if contents != []:
            return results.FAILURE
        paths = ['composite_mixin_test_1', 'composite_mixin_test_2']
        for path in paths:
            has_path = (yield self.pathExists(path))
            if has_path:
                return results.FAILURE
        if self.is_list_mkdir:
            yield self.runMkdir(paths)
        else:
            for path in paths:
                yield self.runMkdir(path)
        for path in paths:
            has_path = (yield self.pathExists(path))
            if not has_path:
                return results.FAILURE
        contents = (yield self.runGlob('*'))
        contents.sort()
        for (i, path) in enumerate(paths):
            if not contents[i].endswith(path):
                return results.FAILURE
        if self.is_list_rmdir:
            yield self.runRmdir(paths)
        else:
            for path in paths:
                yield self.runRmdir(path)
        for path in paths:
            has_path = (yield self.pathExists(path))
            if has_path:
                return results.FAILURE
        return results.SUCCESS

class CompositeStepMixinMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, is_list_mkdir=True, is_list_rmdir=True):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(TestCompositeMixinStep(is_list_mkdir=is_list_mkdir, is_list_rmdir=is_list_rmdir))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_compositemixin_rmdir_list(self):
        if False:
            print('Hello World!')
        yield self.do_compositemixin_test(is_list_mkdir=False, is_list_rmdir=True)

    @defer.inlineCallbacks
    def test_compositemixin(self):
        if False:
            return 10
        yield self.do_compositemixin_test(is_list_mkdir=False, is_list_rmdir=False)

    @defer.inlineCallbacks
    def do_compositemixin_test(self, is_list_mkdir, is_list_rmdir):
        if False:
            while True:
                i = 10
        yield self.setup_config(is_list_mkdir=is_list_mkdir, is_list_rmdir=is_list_rmdir)
        change = {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}
        build = (yield self.doForceBuild(wantSteps=True, useChange=change, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['results'], results.SUCCESS)

class CompositeStepMixinMasterPb(CompositeStepMixinMaster):
    proto = 'pb'

class CompositeStepMixinMasterMsgPack(CompositeStepMixinMaster):
    proto = 'msgpack'

    @defer.inlineCallbacks
    def test_compositemixin_mkdir_rmdir_lists(self):
        if False:
            return 10
        yield self.do_compositemixin_test(is_list_mkdir=True, is_list_rmdir=True)