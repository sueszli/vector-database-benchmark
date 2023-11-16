from twisted.internet import defer
from buildbot.worker.protocols import base

class FakeTrivialConnection(base.Connection):
    info = {}

    def __init__(self):
        if False:
            return 10
        super().__init__('Fake')

    def loseConnection(self):
        if False:
            while True:
                i = 10
        self.notifyDisconnected()

    def remoteSetBuilderList(self, builders):
        if False:
            return 10
        return defer.succeed(None)

class FakeConnection(base.Connection):

    def __init__(self, worker):
        if False:
            return 10
        super().__init__(worker.workername)
        self._connected = True
        self.remoteCalls = []
        self.builders = {}
        self.info = {'worker_commands': [], 'version': '0.9.0', 'basedir': '/w', 'system': 'nt'}

    def loseConnection(self):
        if False:
            i = 10
            return i + 15
        self.notifyDisconnected()

    def remotePrint(self, message):
        if False:
            while True:
                i = 10
        self.remoteCalls.append(('remotePrint', message))
        return defer.succeed(None)

    def remoteGetWorkerInfo(self):
        if False:
            while True:
                i = 10
        self.remoteCalls.append(('remoteGetWorkerInfo',))
        return defer.succeed(self.info)

    def remoteSetBuilderList(self, builders):
        if False:
            while True:
                i = 10
        self.remoteCalls.append(('remoteSetBuilderList', builders[:]))
        self.builders = dict(((b, False) for b in builders))
        return defer.succeed(None)

    def remoteStartCommand(self, remoteCommand, builderName, commandId, commandName, args):
        if False:
            while True:
                i = 10
        self.remoteCalls.append(('remoteStartCommand', remoteCommand, builderName, commandId, commandName, args))
        return defer.succeed(None)

    def remoteShutdown(self):
        if False:
            i = 10
            return i + 15
        self.remoteCalls.append(('remoteShutdown',))
        return defer.succeed(None)

    def remoteStartBuild(self, builderName):
        if False:
            print('Hello World!')
        self.remoteCalls.append(('remoteStartBuild', builderName))
        return defer.succeed(None)

    def remoteInterruptCommand(self, builderName, commandId, why):
        if False:
            while True:
                i = 10
        self.remoteCalls.append(('remoteInterruptCommand', builderName, commandId, why))
        return defer.succeed(None)

    def get_peer(self):
        if False:
            return 10
        if self._connected:
            return 'fake_peer'
        return None