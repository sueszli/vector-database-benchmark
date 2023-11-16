import json
import os
import textwrap
from unittest import mock
from autobahn.wamp.exception import TransportLost
from autobahn.wamp.types import SubscribeOptions
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.mq import wamp
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.wamp import connector

class FakeEventDetails:

    def __init__(self, topic):
        if False:
            i = 10
            return i + 15
        self.topic = topic

class ComparableSubscribeOptions(SubscribeOptions):

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, SubscribeOptions):
            return False
        return self.match == other.match
    __repr__ = SubscribeOptions.__str__

class FakeSubscription:

    def __init__(self):
        if False:
            return 10
        self.exception_on_unsubscribe = None

    def unsubscribe(self):
        if False:
            print('Hello World!')
        if self.exception_on_unsubscribe is not None:
            raise self.exception_on_unsubscribe()

class TestException(Exception):
    pass

class FakeWampConnector:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.subscriptions = []

    def topic_match(self, topic):
        if False:
            for i in range(10):
                print('nop')
        topic = topic.split('.')
        owntopic = self.topic.split('.')
        if len(topic) != len(owntopic):
            return False
        for (i, itopic) in enumerate(topic):
            if owntopic[i] != '' and itopic != owntopic[i]:
                return False
        return True

    def subscribe(self, callback, topic=None, options=None):
        if False:
            return 10
        self.topic = topic
        self.qref_cb = callback
        subs = FakeSubscription()
        self.subscriptions.append(subs)
        return subs

    def publish(self, topic, data, options=None):
        if False:
            while True:
                i = 10
        assert self.topic_match(topic)
        self.last_data = data
        details = FakeEventDetails(topic=topic)
        self.qref_cb(json.loads(json.dumps(data)), details=details)

class TopicMatch(unittest.TestCase):

    def test_topic_match(self):
        if False:
            print('Hello World!')
        matches = [('a.b.c', 'a.b.c'), ('a..c', 'a.c.c'), ('a.b.', 'a.b.c'), ('.b.', 'a.b.c')]
        for (i, j) in matches:
            w = FakeWampConnector()
            w.topic = i
            self.assertTrue(w.topic_match(j))

    def test_topic_not_match(self):
        if False:
            for i in range(10):
                print('nop')
        matches = [('a.b.c', 'a.b.d'), ('a..c', 'a.b.d'), ('a.b.', 'a.c.c'), ('.b.', 'a.a.c')]
        for (i, j) in matches:
            w = FakeWampConnector()
            w.topic = i
            self.assertFalse(w.topic_match(j))

class WampMQ(TestReactorMixin, unittest.TestCase):
    """
        Stimulate the code with a fake wamp router:
        A router which only accepts one subscriber on one topic
    """

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        self.master.wamp = FakeWampConnector()
        self.mq = wamp.WampMQ()
        yield self.mq.setServiceParent(self.master)
        yield self.mq.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        if self.mq.running:
            yield self.mq.stopService()

    @defer.inlineCallbacks
    def test_startConsuming_basic(self):
        if False:
            return 10
        self.master.wamp.subscribe = mock.Mock()
        yield self.mq.startConsuming(None, ('a', 'b'))
        options = ComparableSubscribeOptions(details_arg='details')
        self.master.wamp.subscribe.assert_called_with(mock.ANY, 'org.buildbot.mq.a.b', options=options)

    @defer.inlineCallbacks
    def test_startConsuming_wildcard(self):
        if False:
            print('Hello World!')
        self.master.wamp.subscribe = mock.Mock()
        yield self.mq.startConsuming(None, ('a', None))
        options = ComparableSubscribeOptions(match='wildcard', details_arg='details')
        self.master.wamp.subscribe.assert_called_with(mock.ANY, 'org.buildbot.mq.a.', options=options)

    @defer.inlineCallbacks
    def test_forward_data(self):
        if False:
            i = 10
            return i + 15
        callback = mock.Mock()
        yield self.mq.startConsuming(callback, ('a', 'b'))
        yield self.mq._produce(('a', 'b'), 'foo')
        callback.assert_called_with(('a', 'b'), 'foo')
        self.assertEqual(self.master.wamp.last_data, 'foo')

    @defer.inlineCallbacks
    def test_unsubscribe_ignores_transport_lost(self):
        if False:
            while True:
                i = 10
        callback = mock.Mock()
        consumer = (yield self.mq.startConsuming(callback, ('a', 'b')))
        self.assertEqual(len(self.master.wamp.subscriptions), 1)
        self.master.wamp.subscriptions[0].exception_on_unsubscribe = TransportLost
        yield consumer.stopConsuming()

    @defer.inlineCallbacks
    def test_unsubscribe_logs_exceptions(self):
        if False:
            return 10
        callback = mock.Mock()
        consumer = (yield self.mq.startConsuming(callback, ('a', 'b')))
        self.assertEqual(len(self.master.wamp.subscriptions), 1)
        self.master.wamp.subscriptions[0].exception_on_unsubscribe = TestException
        yield consumer.stopConsuming()
        self.assertEqual(len(self.flushLoggedErrors(TestException)), 1)

    @defer.inlineCallbacks
    def test_forward_data_wildcard(self):
        if False:
            return 10
        callback = mock.Mock()
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq._produce(('a', 'b'), 'foo')
        callback.assert_called_with(('a', 'b'), 'foo')
        self.assertEqual(self.master.wamp.last_data, 'foo')

    @defer.inlineCallbacks
    def test_waits_for_called_callback(self):
        if False:
            for i in range(10):
                print('nop')

        def callback(_, __):
            if False:
                while True:
                    i = 10
            return defer.succeed(None)
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq._produce(('a', 'b'), 'foo')
        self.assertEqual(self.master.wamp.last_data, 'foo')
        d = self.mq.stopService()
        self.assertTrue(d.called)

    @defer.inlineCallbacks
    def test_waits_for_non_called_callback(self):
        if False:
            return 10
        d1 = defer.Deferred()

        def callback(_, __):
            if False:
                return 10
            return d1
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq._produce(('a', 'b'), 'foo')
        self.assertEqual(self.master.wamp.last_data, 'foo')
        d = self.mq.stopService()
        self.assertFalse(d.called)
        d1.callback(None)
        self.assertTrue(d.called)

class FakeConfig:
    mq = {'type': 'wamp', 'router_url': 'wss://foo', 'realm': 'realm1'}

class WampMQReal(TestReactorMixin, unittest.TestCase):
    """
        Tests a little bit more painful to run, but which involve real communication with
        a wamp router
    """
    HOW_TO_RUN = textwrap.dedent('        define WAMP_ROUTER_URL to a wamp router to run this test\n        > crossbar init\n        > crossbar start &\n        > export WAMP_ROUTER_URL=ws://localhost:8080/ws\n        > trial buildbot.unit.test_mq_wamp')
    timeout = 2

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        if 'WAMP_ROUTER_URL' not in os.environ:
            raise unittest.SkipTest(self.HOW_TO_RUN)
        self.master = fakemaster.make_master(self)
        self.mq = wamp.WampMQ()
        yield self.mq.setServiceParent(self.master)
        self.connector = self.master.wamp = connector.WampConnector()
        yield self.connector.setServiceParent(self.master)
        yield self.master.startService()
        config = FakeConfig()
        config.mq['router_url'] = os.environ['WAMP_ROUTER_URL']
        yield self.connector.reconfigServiceWithBuildbotConfig(config)

    def tearDown(self):
        if False:
            print('Hello World!')
        return self.master.stopService()

    @defer.inlineCallbacks
    def test_forward_data(self):
        if False:
            print('Hello World!')
        d = defer.Deferred()
        callback = mock.Mock(side_effect=lambda *a, **kw: d.callback(None))
        yield self.mq.startConsuming(callback, ('a', 'b'))
        yield self.mq._produce(('a', 'b'), 'foo')
        yield d
        callback.assert_called_with(('a', 'b'), 'foo')

    @defer.inlineCallbacks
    def test_forward_data_wildcard(self):
        if False:
            i = 10
            return i + 15
        d = defer.Deferred()
        callback = mock.Mock(side_effect=lambda *a, **kw: d.callback(None))
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq._produce(('a', 'b'), 'foo')
        yield d
        callback.assert_called_with(('a', 'b'), 'foo')