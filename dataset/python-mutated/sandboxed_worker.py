import subprocess
from twisted.internet import defer
from twisted.internet import protocol
from twisted.internet import reactor
from buildbot.util.service import AsyncService

class WorkerProcessProtocol(protocol.ProcessProtocol):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.finished_deferred = defer.Deferred()

    def outReceived(self, data):
        if False:
            print('Hello World!')
        print(data)

    def errReceived(self, data):
        if False:
            while True:
                i = 10
        print(data)

    def processEnded(self, _):
        if False:
            print('Hello World!')
        self.finished_deferred.callback(None)

    def waitForFinish(self):
        if False:
            print('Hello World!')
        return self.finished_deferred

class SandboxedWorker(AsyncService):

    def __init__(self, masterhost, port, name, passwd, workerdir, sandboxed_worker_path, protocol='pb'):
        if False:
            i = 10
            return i + 15
        self.masterhost = masterhost
        self.port = port
        self.workername = name
        self.workerpasswd = passwd
        self.workerdir = workerdir
        self.sandboxed_worker_path = sandboxed_worker_path
        self.protocol = protocol
        self.worker = None

    def startService(self):
        if False:
            while True:
                i = 10
        res = subprocess.run([self.sandboxed_worker_path, 'create-worker', f'--protocol={self.protocol}', '-q', self.workerdir, self.masterhost + ':' + str(self.port), self.workername, self.workerpasswd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0:
            raise RuntimeError('\n'.join(['Unable to create worker!', res.stdout.decode(), res.stderr.decode()]))
        self.processprotocol = processProtocol = WorkerProcessProtocol()
        args = [self.sandboxed_worker_path, 'start', '--nodaemon', self.workerdir]
        self.process = reactor.spawnProcess(processProtocol, self.sandboxed_worker_path, args=args)
        self.worker = self.master.workers.getWorkerByName(self.workername)
        return super().startService()

    @defer.inlineCallbacks
    def shutdownWorker(self):
        if False:
            return 10
        if self.worker is None:
            return
        yield self.worker.shutdown()
        yield self.processprotocol.waitForFinish()