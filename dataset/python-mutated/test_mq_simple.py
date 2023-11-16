from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.mq import simple
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin

class SimpleMQ(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        self.mq = simple.SimpleMQ()
        self.mq.setServiceParent(self.master)
        yield self.mq.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        if self.mq.running:
            yield self.mq.stopService()

    @defer.inlineCallbacks
    def test_forward_data(self):
        if False:
            for i in range(10):
                print('nop')
        callback = mock.Mock()
        yield self.mq.startConsuming(callback, ('a', 'b'))
        yield self.mq.produce(('a', 'b'), 'foo')
        callback.assert_called_with(('a', 'b'), 'foo')

    @defer.inlineCallbacks
    def test_forward_data_wildcard(self):
        if False:
            print('Hello World!')
        callback = mock.Mock()
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq.produce(('a', 'b'), 'foo')
        callback.assert_called_with(('a', 'b'), 'foo')

    @defer.inlineCallbacks
    def test_waits_for_called_callback(self):
        if False:
            print('Hello World!')

        def callback(_, __):
            if False:
                while True:
                    i = 10
            return defer.succeed(None)
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq.produce(('a', 'b'), 'foo')
        d = self.mq.stopService()
        self.assertTrue(d.called)

    @defer.inlineCallbacks
    def test_waits_for_non_called_callback(self):
        if False:
            print('Hello World!')
        d1 = defer.Deferred()

        def callback(_, __):
            if False:
                return 10
            return d1
        yield self.mq.startConsuming(callback, ('a', None))
        yield self.mq.produce(('a', 'b'), 'foo')
        d = self.mq.stopService()
        self.assertFalse(d.called)
        d1.callback(None)
        self.assertTrue(d.called)