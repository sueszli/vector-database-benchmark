from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.config.master import MasterConfig
from buildbot.process.users import manager
from buildbot.util import service

class FakeUserManager(service.AsyncMultiService):
    pass

class TestUserManager(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.master = mock.Mock()
        self.umm = manager.UserManagerManager(self.master)
        self.umm.startService()
        self.config = MasterConfig()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.umm.stopService()

    @defer.inlineCallbacks
    def test_reconfigServiceWithBuildbotConfig(self):
        if False:
            for i in range(10):
                print('nop')
        um1 = FakeUserManager()
        self.config.user_managers = [um1]
        yield self.umm.reconfigServiceWithBuildbotConfig(self.config)
        self.assertTrue(um1.running)
        self.assertIdentical(um1.master, self.master)
        self.config.user_managers = []
        yield self.umm.reconfigServiceWithBuildbotConfig(self.config)
        self.assertIdentical(um1.master, None)