from twisted.internet import defer
from buildbot.process.properties import Interpolate
from buildbot.test.fake.secrets import FakeSecretStorage
from buildbot.test.util.integration import RunMasterBase

class SecretsConfig(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, use_with=False):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        c['secretsProviders'] = [FakeSecretStorage(secretdict={'foo': 'bar', 'something': 'more'})]
        f = BuildFactory()
        if use_with:
            secrets_list = [('pathA', Interpolate('%(secret:something)s'))]
            with f.withSecrets(secrets_list):
                f.addStep(steps.ShellCommand(command=Interpolate('echo %(secret:foo)s')))
        else:
            f.addSteps([steps.ShellCommand(command=Interpolate('echo %(secret:foo)s'))], withSecrets=[('pathA', Interpolate('%(secret:something)s'))])
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_secret(self):
        if False:
            while True:
                i = 10
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, '<foo>'))
        self.assertTrue(res)

    @defer.inlineCallbacks
    def test_withsecrets(self):
        if False:
            i = 10
            return i + 15
        yield self.setup_config(use_with=True)
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, '<foo>'))
        self.assertTrue(res)