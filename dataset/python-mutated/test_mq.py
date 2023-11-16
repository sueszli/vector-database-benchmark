from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.mq import simple
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import interfaces
from buildbot.test.util import tuplematching

class Tests(interfaces.InterfaceTests):

    def setUp(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def test_empty_produce(self):
        if False:
            return 10
        self.mq.produce(('a', 'b', 'c'), {'x': 1})

    def test_signature_produce(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.mq.produce)
        def produce(self, routingKey, data):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_startConsuming(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.mq.startConsuming)
        def startConsuming(self, callback, filter, persistent_name=None):
            if False:
                while True:
                    i = 10
            pass

    @defer.inlineCallbacks
    def test_signature_stopConsuming(self):
        if False:
            print('Hello World!')
        cons = (yield self.mq.startConsuming(lambda : None, ('a',)))

        @self.assertArgSpecMatches(cons.stopConsuming)
        def stopConsuming(self):
            if False:
                while True:
                    i = 10
            pass

    def test_signature_waitUntilEvent(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.mq.waitUntilEvent)
        def waitUntilEvent(self, filter, check_callback):
            if False:
                for i in range(10):
                    print('nop')
            pass

class RealTests(tuplematching.TupleMatchingMixin, Tests):

    @defer.inlineCallbacks
    def do_test_match(self, routingKey, shouldMatch, filter):
        if False:
            print('Hello World!')
        cb = mock.Mock()
        yield self.mq.startConsuming(cb, filter)
        self.mq.produce(routingKey, 'x')
        self.assertEqual(shouldMatch, cb.call_count == 1)
        if shouldMatch:
            cb.assert_called_once_with(routingKey, 'x')

    @defer.inlineCallbacks
    def test_stopConsuming(self):
        if False:
            return 10
        cb = mock.Mock()
        qref = (yield self.mq.startConsuming(cb, ('abc',)))
        self.mq.produce(('abc',), {'x': 1})
        qref.stopConsuming()
        self.mq.produce(('abc',), {'x': 1})
        cb.assert_called_once_with(('abc',), {'x': 1})

    @defer.inlineCallbacks
    def test_stopConsuming_twice(self):
        if False:
            while True:
                i = 10
        cb = mock.Mock()
        qref = (yield self.mq.startConsuming(cb, ('abc',)))
        qref.stopConsuming()
        qref.stopConsuming()

    @defer.inlineCallbacks
    def test_non_persistent(self):
        if False:
            return 10
        cb = mock.Mock()
        qref = (yield self.mq.startConsuming(cb, ('abc',)))
        cb2 = mock.Mock()
        qref2 = (yield self.mq.startConsuming(cb2, ('abc',)))
        qref.stopConsuming()
        self.mq.produce(('abc',), '{}')
        qref = (yield self.mq.startConsuming(cb, ('abc',)))
        qref.stopConsuming()
        qref2.stopConsuming()
        self.assertTrue(cb2.called)
        self.assertFalse(cb.called)

    @defer.inlineCallbacks
    def test_persistent(self):
        if False:
            i = 10
            return i + 15
        cb = mock.Mock()
        qref = (yield self.mq.startConsuming(cb, ('abc',), persistent_name='ABC'))
        qref.stopConsuming()
        self.mq.produce(('abc',), '{}')
        qref = (yield self.mq.startConsuming(cb, ('abc',), persistent_name='ABC'))
        qref.stopConsuming()
        self.assertTrue(cb.called)

    @defer.inlineCallbacks
    def test_waitUntilEvent_check_false(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.mq.waitUntilEvent(('abc',), lambda : False)
        self.assertEqual(d.called, False)
        self.mq.produce(('abc',), {'x': 1})
        self.assertEqual(d.called, True)
        res = (yield d)
        self.assertEqual(res, (('abc',), {'x': 1}))
    timeout = 3

class TestFakeMQ(TestReactorMixin, unittest.TestCase, Tests):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantMq=True)
        self.mq = self.master.mq
        self.mq.verifyMessages = False

class TestSimpleMQ(TestReactorMixin, unittest.TestCase, RealTests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        self.mq = simple.SimpleMQ()
        yield self.mq.setServiceParent(self.master)