from unittest import mock
from parameterized import parameterized
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.util import service
from buildbot.wamp import connector

class FakeConfig:

    def __init__(self, mq_dict):
        if False:
            return 10
        self.mq = mq_dict

class FakeService(service.AsyncMultiService):
    name = 'fakeWampService'

    def __init__(self, url, realm, make, extra=None, debug=False, debug_wamp=False, debug_app=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.make = make
        self.extra = extra

    def gotConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.make(None)
        r = self.make(self)
        r.publish = mock.Mock(spec=r.publish)
        r.register = mock.Mock(spec=r.register)
        r.subscribe = mock.Mock(spec=r.subscribe)
        r.onJoin(None)

class TestedWampConnector(connector.WampConnector):
    serviceClass = FakeService

class WampConnector(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        master = fakemaster.make_master(self)
        self.connector = TestedWampConnector()
        config = FakeConfig({'type': 'wamp', 'router_url': 'wss://foo', 'realm': 'bb'})
        yield self.connector.setServiceParent(master)
        yield master.startService()
        yield self.connector.reconfigServiceWithBuildbotConfig(config)

    @defer.inlineCallbacks
    def test_reconfig_same_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = FakeConfig({'type': 'wamp', 'router_url': 'wss://foo', 'realm': 'bb'})
        yield self.connector.reconfigServiceWithBuildbotConfig(config)

    @parameterized.expand([('type', 'simple'), ('router_url', 'wss://other-foo'), ('realm', 'bb-other'), ('wamp_debug_level', 'info')])
    @defer.inlineCallbacks
    def test_reconfig_does_not_allow_config_change(self, attr_name, attr_value):
        if False:
            while True:
                i = 10
        mq_dict = {'type': 'wamp', 'router_url': 'wss://foo', 'realm': 'bb'}
        mq_dict[attr_name] = attr_value
        with self.assertRaises(ValueError, msg='Cannot use different wamp settings when reconfiguring'):
            yield self.connector.reconfigServiceWithBuildbotConfig(FakeConfig(mq_dict))

    @defer.inlineCallbacks
    def test_startup(self):
        if False:
            while True:
                i = 10
        d = self.connector.getService()
        self.connector.app.gotConnection()
        yield d
        self.connector.service.publish.assert_called_with('org.buildbot.824.connected')

    @defer.inlineCallbacks
    def test_subscribe(self):
        if False:
            i = 10
            return i + 15
        d = self.connector.subscribe('callback', 'topic', 'options')
        self.connector.app.gotConnection()
        yield d
        self.connector.service.subscribe.assert_called_with('callback', 'topic', 'options')

    @defer.inlineCallbacks
    def test_publish(self):
        if False:
            i = 10
            return i + 15
        d = self.connector.publish('topic', 'data', 'options')
        self.connector.app.gotConnection()
        yield d
        self.connector.service.publish.assert_called_with('topic', 'data', options='options')

    @defer.inlineCallbacks
    def test_OnLeave(self):
        if False:
            i = 10
            return i + 15
        d = self.connector.getService()
        self.connector.app.gotConnection()
        yield d
        self.assertTrue(self.connector.master.running)
        self.connector.service.onLeave(None)
        self.assertFalse(self.connector.master.running)