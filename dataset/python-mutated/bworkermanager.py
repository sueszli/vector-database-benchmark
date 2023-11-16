from twisted.internet import defer
from buildbot.util import service

class FakeWorkerManager(service.AsyncMultiService):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.setName('workers')
        self.registrations = {}
        self.connections = {}
        self.workers = {}

    def register(self, worker):
        if False:
            i = 10
            return i + 15
        workerName = worker.workername
        reg = FakeWorkerRegistration(worker)
        self.registrations[workerName] = reg
        return defer.succeed(reg)

    def _unregister(self, registration):
        if False:
            return 10
        del self.registrations[registration.worker.workername]

    def getWorkerByName(self, workerName):
        if False:
            for i in range(10):
                print('nop')
        return self.registrations[workerName].worker

    def newConnection(self, conn, workerName):
        if False:
            print('Hello World!')
        assert workerName not in self.connections
        self.connections[workerName] = conn
        conn.info = {}
        return defer.succeed(True)

class FakeWorkerRegistration:

    def __init__(self, worker):
        if False:
            return 10
        self.updates = []
        self.unregistered = False
        self.worker = worker

    def getPBPort(self):
        if False:
            i = 10
            return i + 15
        return 1234

    def unregister(self):
        if False:
            print('Hello World!')
        assert not self.unregistered, 'called twice'
        self.unregistered = True
        return defer.succeed(None)

    def update(self, worker_config, global_config):
        if False:
            for i in range(10):
                print('nop')
        if worker_config.workername not in self.updates:
            self.updates.append(worker_config.workername)
        return defer.succeed(None)