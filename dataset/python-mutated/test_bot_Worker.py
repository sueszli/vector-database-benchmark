from __future__ import absolute_import
from __future__ import print_function
import os
import shutil
import socket
from twisted.cred import checkers
from twisted.cred import portal
from twisted.internet import defer
from twisted.internet import reactor
from twisted.spread import pb
from twisted.trial import unittest
from zope.interface import implementer
from buildbot_worker import bot
from buildbot_worker.test.util import misc
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock

class MasterPerspective(pb.Avatar):

    def __init__(self, on_keepalive=None):
        if False:
            return 10
        self.on_keepalive = on_keepalive

    def perspective_keepalive(self):
        if False:
            i = 10
            return i + 15
        if self.on_keepalive:
            (on_keepalive, self.on_keepalive) = (self.on_keepalive, None)
            on_keepalive()

@implementer(portal.IRealm)
class MasterRealm(object):

    def __init__(self, perspective, on_attachment):
        if False:
            while True:
                i = 10
        self.perspective = perspective
        self.on_attachment = on_attachment

    @defer.inlineCallbacks
    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            for i in range(10):
                print('nop')
        assert pb.IPerspective in interfaces
        self.mind = mind
        self.perspective.mind = mind
        if self.on_attachment:
            yield self.on_attachment(mind)
        defer.returnValue((pb.IPerspective, self.perspective, lambda : None))

    def shutdown(self):
        if False:
            while True:
                i = 10
        return self.mind.broker.transport.loseConnection()

class TestWorker(misc.PatcherMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.realm = None
        self.worker = None
        self.listeningport = None
        self.basedir = os.path.abspath('basedir')
        if os.path.exists(self.basedir):
            shutil.rmtree(self.basedir)
        os.makedirs(self.basedir)

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        if self.realm:
            yield self.realm.shutdown()
        if self.worker and self.worker.running:
            yield self.worker.stopService()
        if self.listeningport:
            yield self.listeningport.stopListening()
        if os.path.exists(self.basedir):
            shutil.rmtree(self.basedir)

    def start_master(self, perspective, on_attachment=None):
        if False:
            i = 10
            return i + 15
        self.realm = MasterRealm(perspective, on_attachment)
        p = portal.Portal(self.realm)
        p.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(testy=b'westy'))
        self.listeningport = reactor.listenTCP(0, pb.PBServerFactory(p), interface='127.0.0.1')
        return self.listeningport.getHost().port

    def test_constructor_minimal(self):
        if False:
            return 10
        bot.Worker('mstr', 9010, 'me', 'pwd', '/s', 10, protocol='pb')

    def test_constructor_083_tac(self):
        if False:
            while True:
                i = 10
        'invocation as made from default 0.8.3 tac files'
        bot.Worker('mstr', 9010, 'me', 'pwd', '/s', 10, umask=83, protocol='pb', maxdelay=10)

    def test_constructor_091_tac(self):
        if False:
            return 10
        bot.Worker(None, None, 'me', 'pwd', '/s', 10, connection_string='tcp:host=localhost:port=9010', umask=83, protocol='pb', maxdelay=10)

    def test_constructor_invalid_both_styles(self):
        if False:
            for i in range(10):
                print('nop')
        "Can't instantiate with both host/port and connection string."
        self.assertRaises(AssertionError, bot.Worker, 'mstr', 9010, 'me', 'pwd', '/s', 10, connection_string='tcp:anything')

    def test_constructor_invalid_both_styles_partial(self):
        if False:
            return 10
        self.assertRaises(AssertionError, bot.Worker, 'mstr', None, 'me', 'pwd', '/s', 10, connection_string='tcp:anything')

    def test_constructor_invalid_both_styles_partial2(self):
        if False:
            while True:
                i = 10
        "Can't instantiate with both host/port and connection string."
        self.assertRaises(AssertionError, bot.Worker, None, 9010, None, 'me', 'pwd', '/s', 10, connection_string='tcp:anything')

    def test_constructor_full(self):
        if False:
            i = 10
            return i + 15
        bot.Worker('mstr', 9010, 'me', 'pwd', '/s', 10, umask=83, maxdelay=10, keepaliveTimeout=10, unicode_encoding='utf8', protocol='pb', allow_shutdown=True)

    def test_worker_print(self):
        if False:
            return 10
        d = defer.Deferred()

        def call_print(mind):
            if False:
                for i in range(10):
                    print('nop')
            print_d = mind.callRemote('print', 'Hi, worker.')
            print_d.addCallbacks(d.callback, d.errback)
        persp = MasterPerspective()
        port = self.start_master(persp, on_attachment=call_print)
        self.worker = bot.Worker('127.0.0.1', port, 'testy', 'westy', self.basedir, keepalive=0, umask=18, protocol='pb')
        self.worker.startService()
        return d

    def test_recordHostname_uname(self):
        if False:
            return 10
        self.patch_os_uname(lambda : [0, 'test-hostname.domain.com'])
        self.worker = bot.Worker('127.0.0.1', 9999, 'testy', 'westy', self.basedir, keepalive=0, umask=18, protocol='pb')
        self.worker.recordHostname(self.basedir)
        with open(os.path.join(self.basedir, 'twistd.hostname')) as f:
            twistdHostname = f.read().strip()
        self.assertEqual(twistdHostname, 'test-hostname.domain.com')

    def test_recordHostname_getfqdn(self):
        if False:
            print('Hello World!')

        def missing():
            if False:
                print('Hello World!')
            raise AttributeError
        self.patch_os_uname(missing)
        self.patch(socket, 'getfqdn', lambda : 'test-hostname.domain.com')
        self.worker = bot.Worker('127.0.0.1', 9999, 'testy', 'westy', self.basedir, keepalive=0, umask=18, protocol='pb')
        self.worker.recordHostname(self.basedir)
        with open(os.path.join(self.basedir, 'twistd.hostname')) as f:
            twistdHostname = f.read().strip()
        self.assertEqual(twistdHostname, 'test-hostname.domain.com')

    def test_worker_graceful_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that running the build worker's gracefulShutdown method results\n        in a call to the master's shutdown method"
        d = defer.Deferred()
        fakepersp = Mock()
        called = []

        def fakeCallRemote(*args):
            if False:
                while True:
                    i = 10
            called.append(args)
            d1 = defer.succeed(None)
            return d1
        fakepersp.callRemote = fakeCallRemote

        def call_shutdown(mind):
            if False:
                print('Hello World!')
            self.worker.bf.perspective = fakepersp
            shutdown_d = self.worker.gracefulShutdown()
            shutdown_d.addCallbacks(d.callback, d.errback)
        persp = MasterPerspective()
        port = self.start_master(persp, on_attachment=call_shutdown)
        self.worker = bot.Worker('127.0.0.1', port, 'testy', 'westy', self.basedir, keepalive=0, umask=18, protocol='pb')
        self.worker.startService()

        def check(ign):
            if False:
                i = 10
                return i + 15
            self.assertEqual(called, [('shutdown',)])
        d.addCallback(check)
        return d

    def test_worker_shutdown(self):
        if False:
            i = 10
            return i + 15
        'Test watching an existing shutdown_file results in gracefulShutdown\n        being called.'
        worker = bot.Worker('127.0.0.1', 1234, 'testy', 'westy', self.basedir, keepalive=0, umask=18, protocol='pb', allow_shutdown='file')
        worker.gracefulShutdown = Mock()
        exists = Mock()
        mtime = Mock()
        self.patch(os.path, 'exists', exists)
        self.patch(os.path, 'getmtime', mtime)
        mtime.return_value = 0
        exists.return_value = False
        worker._checkShutdownFile()
        self.assertEqual(worker.gracefulShutdown.call_count, 0)
        exists.return_value = True
        mtime.return_value = 2
        worker._checkShutdownFile()
        self.assertEqual(worker.gracefulShutdown.call_count, 1)
        mtime.return_value = 3
        worker._checkShutdownFile()
        self.assertEqual(worker.gracefulShutdown.call_count, 2)
        worker._checkShutdownFile()
        self.assertEqual(worker.gracefulShutdown.call_count, 2)