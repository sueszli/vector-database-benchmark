from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process.properties import Secret
from buildbot.secrets.manager import SecretManager
from buildbot.test.fake import fakemaster
from buildbot.test.fake.secrets import FakeSecretStorage
from buildbot.test.reactor import TestReactorMixin
from buildbot.util.service import BuildbotService

class FakeServiceUsingSecrets(BuildbotService):
    name = 'FakeServiceUsingSecrets'
    secrets = ['foo', 'bar', 'secret']

    def reconfigService(self, foo=None, bar=None, secret=None, other=None):
        if False:
            i = 10
            return i + 15
        self.foo = foo
        self.bar = bar
        self.secret = secret

    def returnRenderedSecrets(self, secretKey):
        if False:
            return 10
        return getattr(self, secretKey)

class TestRenderSecrets(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        fakeStorageService = FakeSecretStorage(secretdict={'foo': 'bar', 'other': 'value'})
        self.secretsrv = SecretManager()
        self.secretsrv.services = [fakeStorageService]
        yield self.secretsrv.setServiceParent(self.master)
        self.srvtest = FakeServiceUsingSecrets()
        yield self.srvtest.setServiceParent(self.master)
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.master.stopService()

    @defer.inlineCallbacks
    def test_secret_rendered(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.srvtest.configureService()
        new = FakeServiceUsingSecrets(foo=Secret('foo'), other=Secret('other'))
        yield self.srvtest.reconfigServiceWithSibling(new)
        self.assertEqual('bar', self.srvtest.returnRenderedSecrets('foo'))

    @defer.inlineCallbacks
    def test_secret_rendered_not_found(self):
        if False:
            print('Hello World!')
        new = FakeServiceUsingSecrets(foo=Secret('foo'))
        yield self.srvtest.reconfigServiceWithSibling(new)
        with self.assertRaises(Exception):
            self.srvtest.returnRenderedSecrets('more')