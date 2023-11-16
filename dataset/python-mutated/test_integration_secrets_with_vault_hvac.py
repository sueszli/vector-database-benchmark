import base64
import subprocess
import time
from unittest.case import SkipTest
from parameterized import parameterized
from twisted.internet import defer
from buildbot.process.properties import Interpolate
from buildbot.secrets.providers.vault_hvac import HashiCorpVaultKvSecretProvider
from buildbot.secrets.providers.vault_hvac import VaultAuthenticatorToken
from buildbot.steps.shell import ShellCommand
from buildbot.test.util.decorators import skipUnlessPlatformIs
from buildbot.test.util.integration import RunMasterBase

@skipUnlessPlatformIs('posix')
class TestVaultHvac(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self, secret_specifier):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        c['secretsProviders'] = [HashiCorpVaultKvSecretProvider(authenticator=VaultAuthenticatorToken('my_vaulttoken'), vault_server='http://localhost:8200', secrets_mount='secret')]
        f = BuildFactory()
        f.addStep(ShellCommand(command=Interpolate(f'echo {secret_specifier} | base64')))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    def start_container(self, image_tag):
        if False:
            while True:
                i = 10
        try:
            image = f'vault:{image_tag}'
            subprocess.check_call(['docker', 'pull', image])
            subprocess.check_call(['docker', 'run', '-d', '-e', 'SKIP_SETCAP=yes', '-e', 'VAULT_DEV_ROOT_TOKEN_ID=my_vaulttoken', '-e', 'VAULT_TOKEN=my_vaulttoken', '--name=vault_for_buildbot', '-p', '8200:8200', image])
            time.sleep(1)
            self.addCleanup(self.remove_container)
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/key', 'value=word'])
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/anykey', 'anyvalue=anyword'])
            subprocess.check_call(['docker', 'exec', '-e', 'VAULT_ADDR=http://127.0.0.1:8200/', 'vault_for_buildbot', 'vault', 'kv', 'put', 'secret/key1/key2', 'id=val'])
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise SkipTest('Vault integration needs docker environment to be setup') from e

    def remove_container(self):
        if False:
            print('Hello World!')
        subprocess.call(['docker', 'rm', '-f', 'vault_for_buildbot'])

    @defer.inlineCallbacks
    def do_secret_test(self, image_tag, secret_specifier, expected_obfuscation, expected_value):
        if False:
            return 10
        self.start_container(image_tag)
        yield self.setup_config(secret_specifier=secret_specifier)
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        patterns = [f'echo {expected_obfuscation}', base64.b64encode((expected_value + '\n').encode('utf-8')).decode('utf-8')]
        res = (yield self.checkBuildStepLogExist(build, patterns))
        self.assertTrue(res)
    all_tags = [('1.9.7',), ('1.10.5',), ('1.11.1',)]

    @parameterized.expand(all_tags)
    @defer.inlineCallbacks
    def test_key(self, image_tag):
        if False:
            while True:
                i = 10
        yield self.do_secret_test(image_tag, '%(secret:key|value)s', '<key|value>', 'word')

    @parameterized.expand(all_tags)
    @defer.inlineCallbacks
    def test_key_any_value(self, image_tag):
        if False:
            return 10
        yield self.do_secret_test(image_tag, '%(secret:anykey|anyvalue)s', '<anykey|anyvalue>', 'anyword')

    @parameterized.expand(all_tags)
    @defer.inlineCallbacks
    def test_nested_key(self, image_tag):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_secret_test(image_tag, '%(secret:key1/key2|id)s', '<key1/key2|id>', 'val')