import os
from twisted.cred import credentials
from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet.endpoints import clientFromString
from twisted.python import log
from twisted.python import util
from twisted.spread import pb
from twisted.trial import unittest
import buildbot
from buildbot import config
from buildbot import worker
from buildbot.process import botmaster
from buildbot.process import builder
from buildbot.process import factory
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.util.eventual import eventually
from buildbot.worker import manager as workermanager
from buildbot.worker.protocols.manager.pb import PBManager
PKI_DIR = util.sibpath(__file__, 'pki')

class FakeWorkerForBuilder(pb.Referenceable):
    """
    Fake worker-side WorkerForBuilder object
    """

class FakeWorkerWorker(pb.Referenceable):
    """
    Fake worker-side Worker object

    @ivar master_persp: remote perspective on the master
    """

    def __init__(self, callWhenBuilderListSet):
        if False:
            return 10
        self.callWhenBuilderListSet = callWhenBuilderListSet
        self.master_persp = None
        self._detach_deferreds = []
        self._detached = False

    def waitForDetach(self):
        if False:
            while True:
                i = 10
        if self._detached:
            return defer.succeed(None)
        d = defer.Deferred()
        self._detach_deferreds.append(d)
        return d

    def setMasterPerspective(self, persp):
        if False:
            for i in range(10):
                print('nop')
        self.master_persp = persp

        def clear_persp():
            if False:
                i = 10
                return i + 15
            self.master_persp = None
        persp.broker.notifyOnDisconnect(clear_persp)

        def fire_deferreds():
            if False:
                print('Hello World!')
            self._detached = True
            (self._detach_deferreds, deferreds) = (None, self._detach_deferreds)
            for d in deferreds:
                d.callback(None)
        persp.broker.notifyOnDisconnect(fire_deferreds)

    def remote_print(self, message):
        if False:
            for i in range(10):
                print('nop')
        log.msg(f'WORKER-SIDE: remote_print({repr(message)})')

    def remote_getWorkerInfo(self):
        if False:
            while True:
                i = 10
        return {'info': 'here', 'worker_commands': {'x': 1}, 'numcpus': 1, 'none': None, 'os_release': b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode(), b'\xe3\x83\xaa\xe3\x83\xaa\xe3\x83\xbc\xe3\x82\xb9\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode(): b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode()}

    def remote_getVersion(self):
        if False:
            print('Hello World!')
        return buildbot.version

    def remote_getCommands(self):
        if False:
            while True:
                i = 10
        return {'x': 1}

    def remote_setBuilderList(self, builder_info):
        if False:
            i = 10
            return i + 15
        builder_names = [n for (n, dir) in builder_info]
        slbuilders = [FakeWorkerForBuilder() for n in builder_names]
        eventually(self.callWhenBuilderListSet)
        return dict(zip(builder_names, slbuilders))

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
            i = 10
            return i + 15
        return defer.succeed(None)

class MyWorker(worker.Worker):

    def attached(self, conn):
        if False:
            i = 10
            return i + 15
        self.detach_d = defer.Deferred()
        return super().attached(conn)

    def detached(self):
        if False:
            i = 10
            return i + 15
        super().detached()
        (self.detach_d, d) = (None, self.detach_d)
        d.callback(None)

class TestWorkerComm(unittest.TestCase, TestReactorMixin):
    """
    Test handling of connections from workers as integrated with
     - Twisted Spread
     - real TCP connections.
     - PBManager

    @ivar master: fake build master
    @ivar pbamanger: L{PBManager} instance
    @ivar botmaster: L{BotMaster} instance
    @ivar worker: master-side L{Worker} instance
    @ivar workerworker: worker-side L{FakeWorkerWorker} instance
    @ivar port: TCP port to connect to
    @ivar server_connection_string: description string for the server endpoint
    @ivar client_connection_string_tpl: description string template for the client
                                endpoint (expects to passed 'port')
    @ivar endpoint: endpoint controlling the outbound connection
                    from worker to master
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
        self.endpoint = None
        self.broker = None
        self._detach_deferreds = []
        self.patch(botmaster, 'Builder', FakeBuilder)
        self.server_connection_string = 'tcp:0:interface=127.0.0.1'
        self.client_connection_string_tpl = 'tcp:host=127.0.0.1:port={port}'

    def tearDown(self):
        if False:
            return 10
        if self.broker:
            del self.broker
        if self.endpoint:
            del self.endpoint
        deferreds = self._detach_deferreds + [self.pbmanager.stopService(), self.botmaster.stopService(), self.workers.stopService()]
        if self.buildworker and self.buildworker.detach_d:
            deferreds.append(self.buildworker.detach_d)
        return defer.gatherResults(deferreds)

    @defer.inlineCallbacks
    def addWorker(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Create a master-side worker instance and add it to the BotMaster\n\n        @param **kwargs: arguments to pass to the L{Worker} constructor.\n        '
        self.buildworker = MyWorker('testworker', 'pw', **kwargs)
        new_config = self.master.config
        new_config.protocols = {'pb': {'port': self.server_connection_string}}
        new_config.workers = [self.buildworker]
        new_config.builders = [config.BuilderConfig(name='bldr', workername='testworker', factory=factory.BuildFactory())]
        yield self.botmaster.reconfigServiceWithBuildbotConfig(new_config)
        yield self.workers.reconfigServiceWithBuildbotConfig(new_config)
        self.port = self.buildworker.registration.getPBPort()

    def connectWorker(self, waitForBuilderList=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Connect a worker the master via PB\n\n        @param waitForBuilderList: don't return until the setBuilderList has\n        been called\n        @returns: L{FakeWorkerWorker} and a Deferred that will fire when it\n        is detached; via deferred\n        "
        factory = pb.PBClientFactory()
        creds = credentials.UsernamePassword(b'testworker', b'pw')
        setBuilderList_d = defer.Deferred()
        workerworker = FakeWorkerWorker(lambda : setBuilderList_d.callback(None))
        login_d = factory.login(creds, workerworker)

        @login_d.addCallback
        def logged_in(persp):
            if False:
                i = 10
                return i + 15
            workerworker.setMasterPerspective(persp)
            workerworker.detach_d = defer.Deferred()
            persp.broker.notifyOnDisconnect(lambda : workerworker.detach_d.callback(None))
            self._detach_deferreds.append(workerworker.detach_d)
            return workerworker
        self.endpoint = clientFromString(reactor, self.client_connection_string_tpl.format(port=self.port))
        connected_d = self.endpoint.connect(factory)
        dlist = [connected_d, login_d]
        if waitForBuilderList:
            dlist.append(setBuilderList_d)
        d = defer.DeferredList(dlist, consumeErrors=True, fireOnOneErrback=True)
        d.addCallback(lambda _: workerworker)
        return d

    def workerSideDisconnect(self, worker):
        if False:
            print('Hello World!')
        'Disconnect from the worker side'
        worker.master_persp.broker.transport.loseConnection()

    @defer.inlineCallbacks
    def test_connect_disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a single worker connecting and disconnecting.'
        yield self.addWorker()
        worker = (yield self.connectWorker())
        self.workerSideDisconnect(worker)
        yield worker.waitForDetach()

    @defer.inlineCallbacks
    def test_tls_connect_disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        'Test with TLS or SSL endpoint.\n\n        According to the deprecation note for the SSL client endpoint,\n        the TLS endpoint is supported from Twistd 16.0.\n\n        TODO add certificate verification (also will require some conditionals\n        on various versions, including PyOpenSSL, service_identity. The CA used\n        to generate the testing cert is in ``PKI_DIR/ca``\n        '

        def escape_colon(path):
            if False:
                while True:
                    i = 10
            return path.replace('\\', '/').replace(':', '\\:')
        self.server_connection_string = ('ssl:port=0:certKey={pub}:privateKey={priv}:' + 'interface=127.0.0.1').format(pub=escape_colon(os.path.join(PKI_DIR, '127.0.0.1.crt')), priv=escape_colon(os.path.join(PKI_DIR, '127.0.0.1.key')))
        self.client_connection_string_tpl = 'ssl:host=127.0.0.1:port={port}'
        yield self.addWorker()
        worker = (yield self.connectWorker())
        self.workerSideDisconnect(worker)
        yield worker.waitForDetach()

    @defer.inlineCallbacks
    def test_worker_info(self):
        if False:
            i = 10
            return i + 15
        yield self.addWorker()
        worker = (yield self.connectWorker())
        props = self.buildworker.info
        self.assertEqual(props.getProperty('info'), 'here')
        self.assertEqual(props.getProperty('os_release'), b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode())
        self.assertEqual(props.getProperty(b'\xe3\x83\xaa\xe3\x83\xaa\xe3\x83\xbc\xe3\x82\xb9\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode()), b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88'.decode())
        self.assertEqual(props.getProperty('none'), None)
        self.assertEqual(props.getProperty('numcpus'), 1)
        self.workerSideDisconnect(worker)
        yield worker.waitForDetach()

    @defer.inlineCallbacks
    def _test_duplicate_worker(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.addWorker()
        worker1 = (yield self.connectWorker())
        try:
            yield self.connectWorker(waitForBuilderList=False)
            connect_failed = False
        except Exception:
            connect_failed = True
        self.assertTrue(connect_failed)
        self.workerSideDisconnect(worker1)
        yield worker1.waitForDetach()
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    @defer.inlineCallbacks
    def _test_duplicate_worker_old_dead(self):
        if False:
            while True:
                i = 10
        yield self.addWorker()
        worker1 = (yield self.connectWorker())

        def remote_print(message):
            if False:
                return 10
            worker1.master_persp.broker.transport.loseConnection()
            raise pb.PBConnectionLost('fake!')
        worker1.remote_print = remote_print
        worker2 = (yield self.connectWorker())
        self.workerSideDisconnect(worker2)
        yield worker1.waitForDetach()
        self.assertEqual(len(self.flushLoggedErrors(pb.PBConnectionLost)), 1)