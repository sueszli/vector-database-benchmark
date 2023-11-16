from unittest import mock
from twisted.trial import unittest
from buildbot.test.fake import fakeprotocol
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import protocols
from buildbot.worker.protocols import base

class TestFakeConnection(protocols.ConnectionInterfaceTest, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.worker = mock.Mock()
        self.conn = fakeprotocol.FakeConnection(self.worker)

class TestConnection(protocols.ConnectionInterfaceTest, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.worker = mock.Mock()
        self.conn = base.Connection(self.worker.workername)

    def test_notify(self):
        if False:
            i = 10
            return i + 15
        cb = mock.Mock()
        self.conn.notifyOnDisconnect(cb)
        self.assertEqual(cb.call_args_list, [])
        self.conn.notifyDisconnected()
        self.assertNotEqual(cb.call_args_list, [])