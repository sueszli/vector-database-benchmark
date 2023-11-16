"""
password store based provider
"""
import os
from pathlib import Path
from twisted.internet import defer
from buildbot import config
from buildbot.secrets.providers.base import SecretProviderBase
from buildbot.util import runprocess

class SecretInPass(SecretProviderBase):
    """
    secret is stored in a password store
    """
    name = 'SecretInPass'

    def checkPassIsInPath(self):
        if False:
            i = 10
            return i + 15
        if not any(((Path(p) / 'pass').is_file() for p in os.environ['PATH'].split(':'))):
            config.error('pass does not exist in PATH')

    def checkPassDirectoryIsAvailableAndReadable(self, dirname):
        if False:
            while True:
                i = 10
        if not os.access(dirname, os.F_OK):
            config.error(f'directory {dirname} does not exist')

    def checkConfig(self, gpgPassphrase=None, dirname=None):
        if False:
            while True:
                i = 10
        self.checkPassIsInPath()
        if dirname:
            self.checkPassDirectoryIsAvailableAndReadable(dirname)

    def reconfigService(self, gpgPassphrase=None, dirname=None):
        if False:
            for i in range(10):
                print('nop')
        self._env = {**os.environ}
        if gpgPassphrase:
            self._env['PASSWORD_STORE_GPG_OPTS'] = f'--passphrase {gpgPassphrase}'
        if dirname:
            self._env['PASSWORD_STORE_DIR'] = dirname

    @defer.inlineCallbacks
    def get(self, entry):
        if False:
            i = 10
            return i + 15
        "\n        get the value from pass identified by 'entry'\n        "
        try:
            (rc, output) = (yield runprocess.run_process(self.master.reactor, ['pass', entry], env=self._env, collect_stderr=False, stderr_is_error=True))
            if rc != 0:
                return None
            return output.decode('utf-8', 'ignore').splitlines()[0]
        except IOError:
            return None