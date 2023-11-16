from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.mq import base
from buildbot.mq import connector
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.util import service

class FakeMQ(service.ReconfigurableServiceMixin, base.MQBase):
    new_config = 'not_called'

    def reconfigServiceWithBuildbotConfig(self, new_config):
        if False:
            return 10
        self.new_config = new_config
        return defer.succeed(None)

    def produce(self, routingKey, data):
        if False:
            while True:
                i = 10
        pass

    def startConsuming(self, callback, filter, persistent_name=None):
        if False:
            i = 10
            return i + 15
        return defer.succeed(None)

class MQConnector(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        self.mqconfig = self.master.config.mq = {}
        self.conn = connector.MQConnector()
        yield self.conn.setServiceParent(self.master)

    def patchFakeMQ(self, name='fake'):
        if False:
            while True:
                i = 10
        self.patch(connector.MQConnector, 'classes', {name: {'class': 'buildbot.test.unit.test_mq_connector.FakeMQ'}})

    @defer.inlineCallbacks
    def test_setup_unknown_type(self):
        if False:
            while True:
                i = 10
        self.mqconfig['type'] = 'unknown'
        with self.assertRaises(AssertionError):
            yield self.conn.setup()

    @defer.inlineCallbacks
    def test_setup_simple_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.patchFakeMQ(name='simple')
        self.mqconfig['type'] = 'simple'
        yield self.conn.setup()
        self.assertIsInstance(self.conn.impl, FakeMQ)
        self.assertEqual(self.conn.impl.produce, self.conn.produce)
        self.assertEqual(self.conn.impl.startConsuming, self.conn.startConsuming)

    @defer.inlineCallbacks
    def test_reconfigServiceWithBuildbotConfig(self):
        if False:
            i = 10
            return i + 15
        self.patchFakeMQ()
        self.mqconfig['type'] = 'fake'
        self.conn.setup()
        new_config = mock.Mock()
        new_config.mq = {'type': 'fake'}
        yield self.conn.reconfigServiceWithBuildbotConfig(new_config)
        self.assertIdentical(self.conn.impl.new_config, new_config)

    @defer.inlineCallbacks
    def test_reconfigService_change_type(self):
        if False:
            return 10
        self.patchFakeMQ()
        self.mqconfig['type'] = 'fake'
        yield self.conn.setup()
        new_config = mock.Mock()
        new_config.mq = {'type': 'other'}
        try:
            yield self.conn.reconfigServiceWithBuildbotConfig(new_config)
        except AssertionError:
            pass
        else:
            self.fail('should have failed')