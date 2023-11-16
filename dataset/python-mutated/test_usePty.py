from packaging.version import parse as parse_version
from twisted import __version__ as twistedVersion
from twisted.internet import defer
from buildbot.test.util.decorators import skipUnlessPlatformIs
from buildbot.test.util.integration import RunMasterBase

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, usePTY):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='if [ -t 1 ] ; then echo in a terminal; else echo "not a terminal"; fi', usePTY=usePTY))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @skipUnlessPlatformIs('posix')
    @defer.inlineCallbacks
    def test_usePTY(self):
        if False:
            while True:
                i = 10
        yield self.setup_config(usePTY=True)
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, 'in a terminal', onlyStdout=True))
        self.assertTrue(res)
        if parse_version(twistedVersion) < parse_version('17.1.0'):
            self.flushWarnings()

    @skipUnlessPlatformIs('posix')
    @defer.inlineCallbacks
    def test_NOusePTY(self):
        if False:
            print('Hello World!')
        yield self.setup_config(usePTY=False)
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, 'not a terminal', onlyStdout=True))
        self.assertTrue(res)