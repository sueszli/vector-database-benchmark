import os
import shutil
import tempfile
import time
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import util
from twisted.trial import unittest
import buildbot_worker.bot
from buildbot import config
from buildbot import worker
from buildbot.process import botmaster
from buildbot.process import builder
from buildbot.process import factory
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.worker import manager as workermanager
from buildbot.worker.protocols.manager.pb import PBManager
PKI_DIR = util.sibpath(__file__, 'pki')
DEFAULT_PORT = os.environ.get('BUILDBOT_TEST_DEFAULT_PORT', '0')

class FakeBuilder(builder.Builder):

    def attached(self, worker, commands):
        if False:
            i = 10
            return i + 15
        return defer.succeed(None)

    def detached(self, worker):
        if False:
            return 10
        pass

    def getOldestRequestTime(self):
        if False:
            print('Hello World!')
        return 0

    def maybeStartBuild(self):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed(None)

class TestingWorker(buildbot_worker.bot.Worker):
    """Add more introspection and scheduling hooks to the real Worker class.

    @ivar tests_connected: a ``Deferred`` that's called back once the PB
                           connection is operational (``gotPerspective``).
                           Callbacks receive the ``Perspective`` object.
    @ivar tests_disconnected: a ``Deferred`` that's called back upon
                              disconnections.

    yielding these in an inlineCallbacks has the effect to wait on the
    corresponding conditions, actually allowing the services to fulfill them.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.tests_disconnected = defer.Deferred()
        self.tests_connected = defer.Deferred()
        self.tests_login_failed = defer.Deferred()
        self.master_perspective = None
        orig_got_persp = self.bf.gotPerspective
        orig_failed_get_persp = self.bf.failedToGetPerspective

        def gotPerspective(persp):
            if False:
                i = 10
                return i + 15
            orig_got_persp(persp)
            self.master_perspective = persp
            self.tests_connected.callback(persp)
            persp.broker.notifyOnDisconnect(lambda : self.tests_disconnected.callback(None))

        def failedToGetPerspective(why, broker):
            if False:
                while True:
                    i = 10
            orig_failed_get_persp(why, broker)
            self.tests_login_failed.callback((why, broker))
        self.bf.gotPerspective = gotPerspective
        self.bf.failedToGetPerspective = failedToGetPerspective

class TestWorkerConnection(unittest.TestCase, TestReactorMixin):
    """
    Test handling of connections from real worker code

    This is meant primarily to test the worker itself.

    @ivar master: fake build master
    @ivar pbmanager: L{PBManager} instance
    @ivar botmaster: L{BotMaster} instance
    @ivar buildworker: L{worker.Worker} instance
    @ivar port: actual TCP port of the master PB service (fixed after call to
                ``addMasterSideWorker``)
    """

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantMq=True, wantData=True, wantDb=True)
        self.pbmanager = self.master.pbmanager = PBManager()
        yield self.pbmanager.setServiceParent(self.master)
        yield self.master.workers.disownServiceParent()
        self.workers = self.master.workers = workermanager.WorkerManager(self.master)
        yield self.workers.setServiceParent(self.master)
        self.botmaster = botmaster.BotMaster()
        yield self.botmaster.setServiceParent(self.master)
        self.master.botmaster = self.botmaster
        self.master.data.updates.workerConfigured = lambda *a, **k: None
        yield self.master.startService()
        self.buildworker = None
        self.port = None
        self.workerworker = None
        self.patch(botmaster, 'Builder', FakeBuilder)
        self.client_connection_string_tpl = 'tcp:host=127.0.0.1:port={port}'
        self.tmpdirs = set()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            return 10
        for tmp in self.tmpdirs:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
        yield self.pbmanager.stopService()
        yield self.botmaster.stopService()
        yield self.workers.stopService()
        if self.buildworker:
            yield self.buildworker.waitForCompleteShutdown()

    @defer.inlineCallbacks
    def addMasterSideWorker(self, connection_string=f'tcp:{DEFAULT_PORT}:interface=127.0.0.1', name='testworker', password='pw', update_port=True, **kwargs):
        if False:
            return 10
        '\n        Create a master-side worker instance and add it to the BotMaster\n\n        @param **kwargs: arguments to pass to the L{Worker} constructor.\n        '
        self.buildworker = worker.Worker(name, password, **kwargs)
        new_config = self.master.config
        new_config.protocols = {'pb': {'port': connection_string}}
        new_config.workers = [self.buildworker]
        new_config.builders = [config.BuilderConfig(name='bldr', workername='testworker', factory=factory.BuildFactory())]
        yield self.botmaster.reconfigServiceWithBuildbotConfig(new_config)
        yield self.workers.reconfigServiceWithBuildbotConfig(new_config)
        if update_port:
            self.port = self.buildworker.registration.getPBPort()

    def workerSideDisconnect(self, worker):
        if False:
            while True:
                i = 10
        'Disconnect from the worker side\n\n        This seems a good way to simulate a broken connection. Returns a Deferred\n        '
        return worker.bf.disconnect()

    def addWorker(self, connection_string_tpl='tcp:host=127.0.0.1:port={port}', password='pw', name='testworker', keepalive=None):
        if False:
            while True:
                i = 10
        'Add a true Worker object to the services.'
        wdir = tempfile.mkdtemp()
        self.tmpdirs.add(wdir)
        return TestingWorker(None, None, name, password, wdir, keepalive, protocol='pb', connection_string=connection_string_tpl.format(port=self.port))

    @defer.inlineCallbacks
    def test_connect_disconnect(self):
        if False:
            while True:
                i = 10
        yield self.addMasterSideWorker()

        def could_not_connect():
            if False:
                print('Hello World!')
            self.fail('Worker never got connected to master')
        timeout = reactor.callLater(10, could_not_connect)
        worker = self.addWorker()
        yield worker.startService()
        yield worker.tests_connected
        timeout.cancel()
        self.assertTrue('bldr' in worker.bot.builders)
        yield worker.stopService()
        yield worker.tests_disconnected

    @defer.inlineCallbacks
    def test_reconnect_network(self):
        if False:
            i = 10
            return i + 15
        yield self.addMasterSideWorker()

        def could_not_connect():
            if False:
                print('Hello World!')
            self.fail('Worker did not reconnect in time to master')
        worker = self.addWorker('tcp:host=127.0.0.1:port={port}')
        yield worker.startService()
        yield worker.tests_connected
        self.assertTrue('bldr' in worker.bot.builders)
        timeout = reactor.callLater(10, could_not_connect)
        yield self.workerSideDisconnect(worker)
        yield worker.tests_connected
        timeout.cancel()
        yield worker.stopService()
        yield worker.tests_disconnected

    @defer.inlineCallbacks
    def test_applicative_reconnection(self):
        if False:
            for i in range(10):
                print('nop')
        'Test reconnection on PB errors.\n\n        The worker starts with a password that the master does not accept\n        at first, and then the master gets reconfigured to accept it.\n        '
        yield self.addMasterSideWorker()
        worker = self.addWorker(password='pw2')
        yield worker.startService()
        yield worker.tests_login_failed
        self.assertEqual(1, len(self.flushLoggedErrors(UnauthorizedLogin)))

        def could_not_connect():
            if False:
                return 10
            self.fail('Worker did not reconnect in time to master')
        yield self.addMasterSideWorker(password='pw2', update_port=False, connection_string=f'tcp:{self.port}:interface=127.0.0.1')
        timeout = reactor.callLater(10, could_not_connect)
        yield worker.tests_connected
        timeout.cancel()
        self.assertTrue('bldr' in worker.bot.builders)
        yield worker.stopService()
        yield worker.tests_disconnected

    @defer.inlineCallbacks
    def test_pb_keepalive(self):
        if False:
            print('Hello World!')
        'Test applicative (PB) keepalives.\n\n        This works by patching the master to callback a deferred on which the\n        test waits.\n        '

        def perspective_keepalive(Connection_self):
            if False:
                print('Hello World!')
            waiter = worker.keepalive_waiter
            if waiter is not None:
                waiter.callback(time.time())
                worker.keepalive_waiter = None
        from buildbot.worker.protocols.pb import Connection
        self.patch(Connection, 'perspective_keepalive', perspective_keepalive)
        yield self.addMasterSideWorker()
        worker = self.addWorker(keepalive=0.1)
        waiter = worker.keepalive_waiter = defer.Deferred()
        yield worker.startService()
        yield worker.tests_connected
        first = (yield waiter)
        yield worker.bf.currentKeepaliveWaiter
        waiter = worker.keepalive_waiter = defer.Deferred()
        second = (yield waiter)
        yield worker.bf.currentKeepaliveWaiter
        self.assertGreater(second, first)
        self.assertLess(second, first + 1)
        yield worker.stopService()
        yield worker.tests_disconnected