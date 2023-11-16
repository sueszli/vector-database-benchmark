from twisted.internet import defer
from twisted.python import log
from twisted.python.constants import NamedConstant
from twisted.python.constants import Names

class States(Names):
    DETACHED = NamedConstant()
    AVAILABLE = NamedConstant()
    BUILDING = NamedConstant()

class AbstractWorkerForBuilder:

    def __init__(self):
        if False:
            print('Hello World!')
        self.ping_watchers = []
        self.state = None
        self.worker = None
        self.builder_name = None
        self.locks = None

    def __repr__(self):
        if False:
            return 10
        r = ['<', self.__class__.__name__]
        if self.builder_name:
            r.extend([' builder=', repr(self.builder_name)])
        if self.worker:
            r.extend([' worker=', repr(self.worker.workername)])
        r.extend([' state=', self.state.name, '>'])
        return ''.join(r)

    def setBuilder(self, b):
        if False:
            for i in range(10):
                print('nop')
        self.builder = b
        self.builder_name = b.name

    def getWorkerCommandVersion(self, command, oldversion=None):
        if False:
            while True:
                i = 10
        if self.remoteCommands is None:
            return oldversion
        return self.remoteCommands.get(command)

    def isAvailable(self):
        if False:
            return 10
        if self.isBusy():
            return False
        if self.worker:
            return self.worker.canStartBuild()
        return False

    def isBusy(self):
        if False:
            while True:
                i = 10
        return self.state != States.AVAILABLE

    def buildStarted(self):
        if False:
            while True:
                i = 10
        self.state = States.BUILDING
        self.worker.buildStarted(self)

    def buildFinished(self):
        if False:
            print('Hello World!')
        self.state = States.AVAILABLE
        if self.worker:
            self.worker.buildFinished(self)

    @defer.inlineCallbacks
    def attached(self, worker, commands):
        if False:
            while True:
                i = 10
        "\n        @type  worker: L{buildbot.worker.Worker}\n        @param worker: the Worker that represents the worker as a whole\n        @type  commands: dict: string -> string, or None\n        @param commands: provides the worker's version of each RemoteCommand\n        "
        self.remoteCommands = commands
        if self.worker is None:
            self.worker = worker
            self.worker.addWorkerForBuilder(self)
        else:
            assert self.worker == worker
        log.msg(f'Worker {worker.workername} attached to {self.builder_name}')
        yield self.worker.conn.remotePrint(message='attached')

    def substantiate_if_needed(self, build):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed(True)

    def insubstantiate_if_needed(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ping(self):
        if False:
            while True:
                i = 10
        'Ping the worker to make sure it is still there. Returns a Deferred\n        that fires with True if it is.\n        '
        newping = not self.ping_watchers
        d = defer.Deferred()
        self.ping_watchers.append(d)
        if newping:
            Ping().ping(self.worker.conn).addBoth(self._pong)
        return d

    def abortPingIfAny(self):
        if False:
            for i in range(10):
                print('nop')
        (watchers, self.ping_watchers) = (self.ping_watchers, [])
        for d in watchers:
            d.errback(PingException('aborted ping'))

    def _pong(self, res):
        if False:
            i = 10
            return i + 15
        (watchers, self.ping_watchers) = (self.ping_watchers, [])
        for d in watchers:
            d.callback(res)

    def detached(self):
        if False:
            return 10
        log.msg(f'Worker {self.worker.workername} detached from {self.builder_name}')
        if self.worker:
            self.worker.removeWorkerForBuilder(self)
        self.worker = None
        self.remoteCommands = None

class PingException(Exception):
    pass

class Ping:
    running = False

    def ping(self, conn):
        if False:
            i = 10
            return i + 15
        assert not self.running
        if not conn:
            return defer.fail(PingException('Worker not connected?'))
        self.running = True
        log.msg('sending ping')
        self.d = defer.Deferred()
        conn.remotePrint(message='ping').addCallbacks(self._pong, self._ping_failed, errbackArgs=(conn,))
        return self.d

    def _pong(self, res):
        if False:
            print('Hello World!')
        log.msg('ping finished: success')
        self.d.callback(True)

    def _ping_failed(self, res, conn):
        if False:
            i = 10
            return i + 15
        log.msg('ping finished: failure')
        conn.loseConnection()
        self.d.errback(res)

class WorkerForBuilder(AbstractWorkerForBuilder):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.state = States.DETACHED

    @defer.inlineCallbacks
    def attached(self, worker, commands):
        if False:
            i = 10
            return i + 15
        yield super().attached(worker, commands)
        self.state = States.AVAILABLE

    def detached(self):
        if False:
            i = 10
            return i + 15
        super().detached()
        if self.worker:
            self.worker.removeWorkerForBuilder(self)
        self.worker = None
        self.state = States.DETACHED

class LatentWorkerForBuilder(AbstractWorkerForBuilder):

    def __init__(self, worker, builder):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.worker = worker
        self.state = States.AVAILABLE
        self.setBuilder(builder)
        self.worker.addWorkerForBuilder(self)
        log.msg(f'Latent worker {worker.workername} attached to {self.builder_name}')

    def substantiate_if_needed(self, build):
        if False:
            print('Hello World!')
        self.state = States.DETACHED
        d = self.substantiate(build)
        return d

    def insubstantiate_if_needed(self):
        if False:
            print('Hello World!')
        self.worker.insubstantiate()

    def attached(self, worker, commands):
        if False:
            print('Hello World!')
        if self.state == States.DETACHED:
            self.state = States.BUILDING
        return super().attached(worker, commands)

    def substantiate(self, build):
        if False:
            while True:
                i = 10
        return self.worker.substantiate(self, build)

    def ping(self):
        if False:
            i = 10
            return i + 15
        if not self.worker.substantiated:
            return defer.fail(PingException('worker is not substantiated'))
        return super().ping()