"""
Test clean shutdown functionality of the master
"""
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.worker.protocols.manager.base import BaseDispatcher
from buildbot.worker.protocols.manager.base import BaseManager

class FakeMaster:
    initLock = defer.DeferredLock()

    def addService(self, svc):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def master(self):
        if False:
            i = 10
            return i + 15
        return self

class TestDispatcher(BaseDispatcher):

    def __init__(self, config_portstr, portstr):
        if False:
            while True:
                i = 10
        super().__init__(portstr)
        self.start_listening_count = 0
        self.stop_listening_count = 0

    def start_listening_port(self):
        if False:
            while True:
                i = 10

        def stopListening():
            if False:
                for i in range(10):
                    print('nop')
            self.stop_listening_count += 1
        self.start_listening_count += 1
        port = mock.Mock()
        port.stopListening = stopListening
        return port

class TestPort:

    def __init__(self, test):
        if False:
            i = 10
            return i + 15
        self.test = test

    def stopListening(self):
        if False:
            i = 10
            return i + 15
        self.test.stop_listening_count += 1

class TestManagerClass(BaseManager):
    dispatcher_class = TestDispatcher

class TestBaseManager(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.manager = TestManagerClass('test_base_manager')
        yield self.manager.setServiceParent(FakeMaster())

    def assert_equal_registration(self, result, expected):
        if False:
            return 10
        result_users = {key: result[key].users for key in result}
        self.assertEqual(result_users, expected)

    def assert_start_stop_listening_counts(self, disp, start_count, stop_count):
        if False:
            i = 10
            return i + 15
        self.assertEqual(disp.start_listening_count, start_count)
        self.assertEqual(disp.stop_listening_count, stop_count)

    @defer.inlineCallbacks
    def test_repr(self):
        if False:
            print('Hello World!')
        reg = (yield self.manager.register('tcp:port', 'x', 'y', 'pf'))
        self.assertEqual(repr(self.manager.dispatchers['tcp:port']), '<base.BaseDispatcher for x on tcp:port>')
        self.assertEqual(repr(reg), '<base.Registration for x on tcp:port>')

    @defer.inlineCallbacks
    def test_register_before_start_service(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.manager.register('tcp:port', 'user', 'pass', 'pf')
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user': ('pass', 'pf')}})
        disp = self.manager.dispatchers['tcp:port']
        self.assert_start_stop_listening_counts(disp, 0, 0)
        self.assertEqual(len(self.manager.services), 1)
        yield self.manager.startService()
        self.assert_start_stop_listening_counts(disp, 1, 0)
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp, 1, 1)

    @defer.inlineCallbacks
    def test_same_registration_two_times(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.manager.startService()
        yield self.manager.register('tcp:port', 'user', 'pass', 'pf')
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user': ('pass', 'pf')}})
        self.assertEqual(len(self.manager.services), 1)
        disp = self.manager.dispatchers['tcp:port']
        self.assert_start_stop_listening_counts(disp, 1, 0)
        with self.assertRaises(KeyError):
            yield self.manager.register('tcp:port', 'user', 'pass', 'pf')
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp, 1, 1)

    @defer.inlineCallbacks
    def test_register_unregister_register(self):
        if False:
            return 10
        yield self.manager.startService()
        reg = (yield self.manager.register('tcp:port', 'user', 'pass', 'pf'))
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user': ('pass', 'pf')}})
        disp = self.manager.dispatchers['tcp:port']
        self.assert_start_stop_listening_counts(disp, 1, 0)
        reg.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {})
        yield self.manager.register('tcp:port', 'user', 'pass', 'pf')
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user': ('pass', 'pf')}})
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp, 1, 1)

    @defer.inlineCallbacks
    def test_register_unregister_empty_disp_users(self):
        if False:
            while True:
                i = 10
        yield self.manager.startService()
        reg = (yield self.manager.register('tcp:port', 'user', 'pass', 'pf'))
        self.assertEqual(len(self.manager.services), 1)
        expected = {'tcp:port': {'user': ('pass', 'pf')}}
        self.assert_equal_registration(self.manager.dispatchers, expected)
        disp = self.manager.dispatchers['tcp:port']
        reg.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {})
        self.assertEqual(reg.username, None)
        self.assertEqual(len(self.manager.services), 0)
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp, 1, 1)

    @defer.inlineCallbacks
    def test_different_ports_same_users(self):
        if False:
            return 10
        yield self.manager.startService()
        reg1 = (yield self.manager.register('tcp:port1', 'user', 'pass', 'pf'))
        reg2 = (yield self.manager.register('tcp:port2', 'user', 'pass', 'pf'))
        reg3 = (yield self.manager.register('tcp:port3', 'user', 'pass', 'pf'))
        disp1 = self.manager.dispatchers['tcp:port1']
        self.assert_start_stop_listening_counts(disp1, 1, 0)
        disp2 = self.manager.dispatchers['tcp:port2']
        self.assert_start_stop_listening_counts(disp2, 1, 0)
        disp3 = self.manager.dispatchers['tcp:port3']
        self.assert_start_stop_listening_counts(disp3, 1, 0)
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port1': {'user': ('pass', 'pf')}, 'tcp:port2': {'user': ('pass', 'pf')}, 'tcp:port3': {'user': ('pass', 'pf')}})
        self.assertEqual(len(self.manager.services), 3)
        yield reg1.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port2': {'user': ('pass', 'pf')}, 'tcp:port3': {'user': ('pass', 'pf')}})
        self.assertEqual(reg1.username, None)
        self.assertEqual(len(self.manager.services), 2)
        self.assert_start_stop_listening_counts(disp1, 1, 1)
        yield reg2.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port3': {'user': ('pass', 'pf')}})
        self.assertEqual(reg2.username, None)
        self.assertEqual(len(self.manager.services), 1)
        self.assert_start_stop_listening_counts(disp2, 1, 1)
        yield reg3.unregister()
        expected = {}
        self.assert_equal_registration(self.manager.dispatchers, expected)
        self.assertEqual(reg3.username, None)
        self.assertEqual(len(self.manager.services), 0)
        self.assert_start_stop_listening_counts(disp3, 1, 1)
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp1, 1, 1)
        self.assert_start_stop_listening_counts(disp2, 1, 1)
        self.assert_start_stop_listening_counts(disp3, 1, 1)

    @defer.inlineCallbacks
    def test_same_port_different_users(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.manager.startService()
        reg1 = (yield self.manager.register('tcp:port', 'user1', 'pass1', 'pf1'))
        reg2 = (yield self.manager.register('tcp:port', 'user2', 'pass2', 'pf2'))
        reg3 = (yield self.manager.register('tcp:port', 'user3', 'pass3', 'pf3'))
        disp = self.manager.dispatchers['tcp:port']
        self.assertEqual(len(self.manager.services), 1)
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user1': ('pass1', 'pf1'), 'user2': ('pass2', 'pf2'), 'user3': ('pass3', 'pf3')}})
        self.assertEqual(len(self.manager.services), 1)
        yield reg1.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user2': ('pass2', 'pf2'), 'user3': ('pass3', 'pf3')}})
        self.assertEqual(reg1.username, None)
        self.assertEqual(len(self.manager.services), 1)
        yield reg2.unregister()
        self.assert_equal_registration(self.manager.dispatchers, {'tcp:port': {'user3': ('pass3', 'pf3')}})
        self.assertEqual(reg2.username, None)
        self.assertEqual(len(self.manager.services), 1)
        yield reg3.unregister()
        expected = {}
        self.assert_equal_registration(self.manager.dispatchers, expected)
        self.assertEqual(reg3.username, None)
        self.assertEqual(len(self.manager.services), 0)
        yield self.manager.stopService()
        self.assert_start_stop_listening_counts(disp, 1, 1)