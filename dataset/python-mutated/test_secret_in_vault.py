import warnings
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.secrets.providers.vault import HashiCorpVaultSecretProvider
from buildbot.test.fake import fakemaster
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.config import ConfigErrorsMixin

class TestSecretInVaultHttpFakeBase(ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self, version):
        if False:
            return 10
        warnings.simplefilter('ignore')
        self.setup_test_reactor()
        self.srvcVault = HashiCorpVaultSecretProvider(vaultServer='http://vaultServer', vaultToken='someToken', apiVersion=version)
        self.master = fakemaster.make_master(self, wantData=True)
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'http://vaultServer', headers={'X-Vault-Token': 'someToken'}))
        yield self.srvcVault.setServiceParent(self.master)
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            return 10
        yield self.srvcVault.stopService()

class TestSecretInVaultV1(TestSecretInVaultHttpFakeBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp(version=1)

    @defer.inlineCallbacks
    def testGetValue(self):
        if False:
            return 10
        self._http.expect(method='get', ep='/v1/secret/value', params=None, data=None, json=None, code=200, content_json={'data': {'value': 'value1'}})
        value = (yield self.srvcVault.get('value'))
        self.assertEqual(value, 'value1')

    @defer.inlineCallbacks
    def test_get_any_key_without_value_name(self):
        if False:
            for i in range(10):
                print('nop')
        self._http.expect(method='get', ep='/v1/secret/any_key', params=None, data=None, json=None, code=200, content_json={'data': {'any_value': 'value1'}})
        yield self.assertFailure(self.srvcVault.get('any_key'), KeyError)

    @defer.inlineCallbacks
    def test_get_any_key_with_value_name(self):
        if False:
            return 10
        self._http.expect(method='get', ep='/v1/secret/any_key', params=None, data=None, json=None, code=200, content_json={'data': {'any_value': 'value1'}})
        value = (yield self.srvcVault.get('any_key/any_value'))
        self.assertEqual(value, 'value1')

    @defer.inlineCallbacks
    def testGetValueNotFound(self):
        if False:
            while True:
                i = 10
        self._http.expect(method='get', ep='/v1/secret/value', params=None, data=None, json=None, code=200, content_json={'data': {'valueNotFound': 'value1'}})
        yield self.assertFailure(self.srvcVault.get('value'), KeyError)

    @defer.inlineCallbacks
    def testGetError(self):
        if False:
            i = 10
            return i + 15
        self._http.expect(method='get', ep='/v1/secret/valueNotFound', params=None, data=None, json=None, code=404, content_json={'data': {'valueNotFound': 'value1'}})
        yield self.assertFailure(self.srvcVault.get('valueNotFound'), KeyError)

    def testCheckConfigSecretInVaultService(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.srvcVault.name, 'SecretInVault')
        self.assertEqual(self.srvcVault.vaultServer, 'http://vaultServer')
        self.assertEqual(self.srvcVault.vaultToken, 'someToken')

    def testCheckConfigErrorSecretInVaultService(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('vaultServer must be a string while it is'):
            self.srvcVault.checkConfig()

    def testCheckConfigErrorSecretInVaultServiceWrongServerAddress(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('vaultToken must be a string while it is'):
            self.srvcVault.checkConfig(vaultServer='serveraddr')

    def test_check_config_error_apiVersion_unsupported(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('apiVersion 0 is not supported'):
            self.srvcVault.checkConfig(vaultServer='serveraddr', vaultToken='vaultToken', apiVersion=0)

    @defer.inlineCallbacks
    def testReconfigSecretInVaultService(self):
        if False:
            return 10
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'serveraddr', headers={'X-Vault-Token': 'someToken'}))
        yield self.srvcVault.reconfigService(vaultServer='serveraddr', vaultToken='someToken')
        self.assertEqual(self.srvcVault.vaultServer, 'serveraddr')
        self.assertEqual(self.srvcVault.vaultToken, 'someToken')

class TestSecretInVaultV2(TestSecretInVaultHttpFakeBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp(version=2)

    @defer.inlineCallbacks
    def testGetValue(self):
        if False:
            return 10
        self._http.expect(method='get', ep='/v1/secret/data/value', params=None, data=None, json=None, code=200, content_json={'data': {'data': {'value': 'value1'}}})
        value = (yield self.srvcVault.get('value'))
        self.assertEqual(value, 'value1')

    @defer.inlineCallbacks
    def test_get_any_key_without_value_name(self):
        if False:
            return 10
        self._http.expect(method='get', ep='/v1/secret/data/any_key', params=None, data=None, json=None, code=200, content_json={'data': {'data': {'any_value': 'value1'}}})
        yield self.assertFailure(self.srvcVault.get('any_key'), KeyError)

    @defer.inlineCallbacks
    def test_get_any_key_with_value_name(self):
        if False:
            i = 10
            return i + 15
        self._http.expect(method='get', ep='/v1/secret/data/any_key', params=None, data=None, json=None, code=200, content_json={'data': {'data': {'any_value': 'value1'}}})
        value = (yield self.srvcVault.get('any_key/any_value'))
        self.assertEqual(value, 'value1')

    @defer.inlineCallbacks
    def testGetValueNotFound(self):
        if False:
            while True:
                i = 10
        self._http.expect(method='get', ep='/v1/secret/data/value', params=None, data=None, json=None, code=200, content_json={'data': {'data': {'valueNotFound': 'value1'}}})
        yield self.assertFailure(self.srvcVault.get('value'), KeyError)

    @defer.inlineCallbacks
    def testGetError(self):
        if False:
            i = 10
            return i + 15
        self._http.expect(method='get', ep='/v1/secret/data/valueNotFound', params=None, data=None, json=None, code=404, content_json={'data': {'data': {'valueNotFound': 'value1'}}})
        yield self.assertFailure(self.srvcVault.get('valueNotFound'), KeyError)