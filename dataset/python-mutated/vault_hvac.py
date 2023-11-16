"""
HVAC based providers
"""
import importlib.metadata
from packaging.version import parse as parse_version
from twisted.internet import defer
from twisted.internet import threads
from buildbot import config
from buildbot.secrets.providers.base import SecretProviderBase

class VaultAuthenticator:
    """
    base HVAC authenticator class
    """

    def authenticate(self, client):
        if False:
            i = 10
            return i + 15
        pass

class VaultAuthenticatorToken(VaultAuthenticator):
    """
    HVAC authenticator for static token
    """

    def __init__(self, token):
        if False:
            return 10
        self.token = token

    def authenticate(self, client):
        if False:
            while True:
                i = 10
        client.token = self.token

class VaultAuthenticatorApprole(VaultAuthenticator):
    """
    HVAC authenticator for Approle login method
    """

    def __init__(self, roleId, secretId):
        if False:
            for i in range(10):
                print('nop')
        self.roleId = roleId
        self.secretId = secretId

    def authenticate(self, client):
        if False:
            while True:
                i = 10
        client.auth.approle.login(role_id=self.roleId, secret_id=self.secretId)

class HashiCorpVaultKvSecretProvider(SecretProviderBase):
    """
    Basic provider where each secret is stored in Vault KV secret engine.
    In case more secret engines are going to be supported, each engine should have it's own class.
    """
    name = 'SecretInVaultKv'

    def checkConfig(self, vault_server=None, authenticator=None, secrets_mount=None, api_version=2, path_delimiter='|', path_escape='\\'):
        if False:
            while True:
                i = 10
        try:
            import hvac
            [hvac]
        except ImportError:
            config.error(f'{self.__class__.__name__} needs the hvac package installed ' + '(pip install hvac)')
        if not isinstance(vault_server, str):
            config.error(f'vault_server must be a string while it is {type(vault_server)}')
        if not isinstance(path_delimiter, str) or len(path_delimiter) > 1:
            config.error('path_delimiter must be a single character')
        if not isinstance(path_escape, str) or len(path_escape) > 1:
            config.error('path_escape must be a single character')
        if not isinstance(authenticator, VaultAuthenticator):
            config.error(f'authenticator must be instance of VaultAuthenticator while it is {type(authenticator)}')
        if api_version not in [1, 2]:
            config.error(f'api_version {api_version} is not supported')

    def reconfigService(self, vault_server=None, authenticator=None, secrets_mount=None, api_version=2, path_delimiter='|', path_escape='\\'):
        if False:
            while True:
                i = 10
        try:
            import hvac
        except ImportError:
            config.error(f'{self.__class__.__name__} needs the hvac package installed ' + '(pip install hvac)')
        if secrets_mount is None:
            secrets_mount = 'secret'
        self.secrets_mount = secrets_mount
        self.path_delimiter = path_delimiter
        self.path_escape = path_escape
        self.authenticator = authenticator
        self.api_version = api_version
        if vault_server.endswith('/'):
            vault_server = vault_server[:-1]
        self.client = hvac.Client(vault_server)
        self.version = parse_version(importlib.metadata.version('hvac'))
        self.client.secrets.kv.default_kv_version = api_version
        return self

    def escaped_split(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        parse and split string, respecting escape characters\n        '
        ret = []
        current = []
        itr = iter(s)
        for ch in itr:
            if ch == self.path_escape:
                try:
                    current.append(next(itr))
                except StopIteration:
                    pass
            elif ch == self.path_delimiter:
                ret.append(''.join(current))
                current = []
            else:
                current.append(ch)
        ret.append(''.join(current))
        return ret

    def thd_hvac_wrap_read(self, path):
        if False:
            for i in range(10):
                print('nop')
        if self.api_version == 1:
            return self.client.secrets.kv.v1.read_secret(path=path, mount_point=self.secrets_mount)
        else:
            if self.version >= parse_version('1.1.1'):
                return self.client.secrets.kv.v2.read_secret_version(path=path, mount_point=self.secrets_mount, raise_on_deleted_version=True)
            return self.client.secrets.kv.v2.read_secret_version(path=path, mount_point=self.secrets_mount)

    def thd_hvac_get(self, path):
        if False:
            while True:
                i = 10
        '\n        query secret from Vault and re-authenticate if not authenticated\n        '
        if not self.client.is_authenticated():
            self.authenticator.authenticate(self.client)
        response = self.thd_hvac_wrap_read(path=path)
        return response

    @defer.inlineCallbacks
    def get(self, entry):
        if False:
            return 10
        '\n        get the value from vault secret backend\n        '
        parts = self.escaped_split(entry)
        if len(parts) == 1:
            raise KeyError(f"Vault secret specification must contain attribute name separated from path by '{self.path_delimiter}'")
        if len(parts) > 2:
            raise KeyError(f"Multiple separators ('{self.path_delimiter}') found in vault path '{entry}'. All occurences of '{self.path_delimiter}' in path or attribute name must be escaped using '{self.path_escape}'")
        name = parts[0]
        key = parts[1]
        response = (yield threads.deferToThread(self.thd_hvac_get, path=name))
        if self.api_version == 2:
            response = response['data']
        try:
            return response['data'][key]
        except KeyError as e:
            raise KeyError(f'The secret {entry} does not exist in Vault provider: {e}') from e