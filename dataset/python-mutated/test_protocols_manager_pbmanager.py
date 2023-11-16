"""
Test clean shutdown functionality of the master
"""
from unittest import mock
from twisted.cred import credentials
from twisted.internet import defer
from twisted.spread import pb
from twisted.trial import unittest
from buildbot.worker.protocols.manager.pb import PBManager

class FakeMaster:
    initLock = defer.DeferredLock()

    def addService(self, svc):
        if False:
            return 10
        pass

    @property
    def master(self):
        if False:
            return 10
        return self

class TestPBManager(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.pbm = PBManager()
        yield self.pbm.setServiceParent(FakeMaster())
        self.pbm.startService()
        self.connections = []

    def tearDown(self):
        if False:
            return 10
        return self.pbm.stopService()

    def perspectiveFactory(self, mind, username):
        if False:
            while True:
                i = 10
        persp = mock.Mock()
        persp.is_my_persp = True
        persp.attached = lambda mind: defer.succeed(None)
        self.connections.append(username)
        return defer.succeed(persp)

    @defer.inlineCallbacks
    def test_register_unregister(self):
        if False:
            return 10
        portstr = 'tcp:0:interface=127.0.0.1'
        reg = (yield self.pbm.register(portstr, 'boris', 'pass', self.perspectiveFactory))
        self.assertIn(portstr, self.pbm.dispatchers)
        disp = self.pbm.dispatchers[portstr]
        self.assertIn('boris', disp.users)
        username = (yield disp.requestAvatarId(credentials.UsernamePassword(b'boris', b'pass')))
        self.assertEqual(username, b'boris')
        avatar = (yield disp.requestAvatar(b'boris', mock.Mock(), pb.IPerspective))
        (_, persp, __) = avatar
        self.assertTrue(persp.is_my_persp)
        self.assertIn('boris', self.connections)
        yield reg.unregister()

    @defer.inlineCallbacks
    def test_register_no_user(self):
        if False:
            while True:
                i = 10
        portstr = 'tcp:0:interface=127.0.0.1'
        reg = (yield self.pbm.register(portstr, 'boris', 'pass', self.perspectiveFactory))
        self.assertIn(portstr, self.pbm.dispatchers)
        disp = self.pbm.dispatchers[portstr]
        self.assertIn('boris', disp.users)
        username = (yield disp.requestAvatarId(credentials.UsernamePassword(b'boris', b'pass')))
        self.assertEqual(username, b'boris')
        with self.assertRaises(ValueError):
            yield disp.requestAvatar(b'notboris', mock.Mock(), pb.IPerspective)
        self.assertNotIn('boris', self.connections)
        yield reg.unregister()

    @defer.inlineCallbacks
    def test_requestAvatarId_noinitLock(self):
        if False:
            return 10
        portstr = 'tcp:0:interface=127.0.0.1'
        reg = (yield self.pbm.register(portstr, 'boris', 'pass', self.perspectiveFactory))
        disp = self.pbm.dispatchers[portstr]
        d = disp.requestAvatarId(credentials.UsernamePassword(b'boris', b'pass'))
        self.assertTrue(d.called, 'requestAvatarId should have been called since the lock is free')
        yield reg.unregister()

    @defer.inlineCallbacks
    def test_requestAvatarId_initLock(self):
        if False:
            i = 10
            return i + 15
        portstr = 'tcp:0:interface=127.0.0.1'
        reg = (yield self.pbm.register(portstr, 'boris', 'pass', self.perspectiveFactory))
        disp = self.pbm.dispatchers[portstr]
        try:
            yield self.pbm.master.initLock.acquire()
            d = disp.requestAvatarId(credentials.UsernamePassword(b'boris', b'pass'))
            self.assertFalse(d.called, 'requestAvatarId should block until the lock is released')
        finally:
            yield self.pbm.master.initLock.release()
        self.assertTrue(d.called, 'requestAvatarId should have been called after the lock was released')
        yield reg.unregister()