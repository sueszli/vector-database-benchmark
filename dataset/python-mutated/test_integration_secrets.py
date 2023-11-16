import os
from parameterized import parameterized
from twisted.internet import defer
from buildbot.process.properties import Interpolate
from buildbot.reporters.http import HttpStatusPush
from buildbot.test.fake.secrets import FakeSecretStorage
from buildbot.test.util.integration import RunMasterBase

class FakeSecretReporter(HttpStatusPush):

    def sendMessage(self, reports):
        if False:
            print('Hello World!')
        assert self.auth == ('user', 'myhttppasswd')
        self.reported = True

class SecretsConfig(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, use_interpolation):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.plugins import util
        from buildbot.process.factory import BuildFactory
        fake_reporter = FakeSecretReporter('http://example.com/hook', auth=('user', Interpolate('%(secret:httppasswd)s')))
        c['services'] = [fake_reporter]
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        c['secretsProviders'] = [FakeSecretStorage(secretdict={'foo': 'secretvalue', 'something': 'more', 'httppasswd': 'myhttppasswd'})]
        f = BuildFactory()
        if use_interpolation:
            if os.name == 'posix':
                command = Interpolate('echo %(secret:foo)s | ' + 'sed "s/secretvalue/The password was there/"')
            else:
                command = Interpolate('echo %(secret:foo)s')
        else:
            command = ['echo', util.Secret('foo')]
        f.addStep(steps.ShellCommand(command=command))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)
        return fake_reporter

    @parameterized.expand([('with_interpolation', True), ('plain_command', False)])
    @defer.inlineCallbacks
    def test_secret(self, name, use_interpolation):
        if False:
            i = 10
            return i + 15
        fake_reporter = (yield self.setup_config(use_interpolation))
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, 'echo <foo>'))
        yield self.checkBuildStepLogExist(build, 'argv:.*echo.*<foo>', regex=True)
        if os.name == 'posix' and use_interpolation:
            res &= (yield self.checkBuildStepLogExist(build, 'The password was there'))
        self.assertTrue(res)
        self.assertNotIn('secretvalue', repr(build))
        self.assertTrue(fake_reporter.reported)

    @parameterized.expand([('with_interpolation', True), ('plain_command', False)])
    @defer.inlineCallbacks
    def test_secretReconfig(self, name, use_interpolation):
        if False:
            print('Hello World!')
        yield self.setup_config(use_interpolation)
        self.master_config_dict['secretsProviders'] = [FakeSecretStorage(secretdict={'foo': 'different_value', 'something': 'more'})]
        yield self.master.reconfig()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, 'echo <foo>'))
        self.assertTrue(res)
        self.assertNotIn('different_value', repr(build))

class SecretsConfigPB(SecretsConfig):
    proto = 'pb'

class SecretsConfigMsgPack(SecretsConfig):
    proto = 'msgpack'