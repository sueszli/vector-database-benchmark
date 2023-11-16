import base64
import subprocess
from unittest.case import SkipTest
from twisted.internet import defer
from buildbot.process.properties import Interpolate
from buildbot.secrets.providers.vault import HashiCorpVaultSecretProvider
from buildbot.steps.shell import ShellCommand
from buildbot.test.util.decorators import skipUnlessPlatformIs
from buildbot.test.util.integration import RunMasterBase
from buildbot.test.util.warnings import assertProducesWarning
from buildbot.warnings import DeprecatedApiWarning

@skipUnlessPlatformIs('posix')
class SecretsConfig(RunMasterBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        try:
            subprocess.check_call(['docker', 'pull', 'vault'])
            subprocess.check_call(['docker', 'run', '-d', '-e', 'SKIP_SETCAP=yes', '-e', 'VAULT_DEV_ROOT_TOKEN_ID=my_vaulttoken', '-e', 'VAULT_TOKEN=my_vaulttoken', '--name=vault_for_buildbot', '-p', '8200:8200', 'vault'])
            self.addCleanup(self.remove_container)
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/key', 'value=word'])
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/anykey', 'anyvalue=anyword'])
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/key1/key2', 'id=val'])
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise SkipTest('Vault integration needs docker environment to be setup') from e

    @defer.inlineCallbacks
    def setup_config(self, secret_specifier):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        c['secretsProviders'] = [HashiCorpVaultSecretProvider(vaultToken='my_vaulttoken', vaultServer='http://localhost:8200', apiVersion=2)]
        f = BuildFactory()
        f.addStep(ShellCommand(command=Interpolate(f'echo {secret_specifier} | base64')))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    def remove_container(self):
        if False:
            return 10
        subprocess.call(['docker', 'rm', '-f', 'vault_for_buildbot'])

    @defer.inlineCallbacks
    def do_secret_test(self, secret_specifier, expected_obfuscation, expected_value):
        if False:
            i = 10
            return i + 15
        with assertProducesWarning(DeprecatedApiWarning):
            yield self.setup_config(secret_specifier=secret_specifier)
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        patterns = [f'echo {expected_obfuscation}', base64.b64encode((expected_value + '\n').encode('utf-8')).decode('utf-8')]
        res = (yield self.checkBuildStepLogExist(build, patterns))
        self.assertTrue(res)

    @defer.inlineCallbacks
    def test_key(self):
        if False:
            i = 10
            return i + 15
        yield self.do_secret_test('%(secret:key)s', '<key>', 'word')

    @defer.inlineCallbacks
    def test_key_value(self):
        if False:
            return 10
        yield self.do_secret_test('%(secret:key/value)s', '<key/value>', 'word')

    @defer.inlineCallbacks
    def test_any_key(self):
        if False:
            return 10
        yield self.do_secret_test('%(secret:anykey/anyvalue)s', '<anykey/anyvalue>', 'anyword')

    @defer.inlineCallbacks
    def test_nested_key(self):
        if False:
            i = 10
            return i + 15
        yield self.do_secret_test('%(secret:key1/key2/id)s', '<key1/key2/id>', 'val')