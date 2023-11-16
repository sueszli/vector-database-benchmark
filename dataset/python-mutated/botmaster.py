from twisted.internet import defer
from buildbot.process import botmaster
from buildbot.util import service

class FakeBotMaster(service.AsyncMultiService, botmaster.LockRetrieverMixin):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.setName('fake-botmaster')
        self.builders = {}
        self.buildsStartedForWorkers = []
        self.delayShutdown = False

    def getBuildersForWorker(self, workername):
        if False:
            for i in range(10):
                print('nop')
        return self.builders.get(workername, [])

    def maybeStartBuildsForWorker(self, workername):
        if False:
            i = 10
            return i + 15
        self.buildsStartedForWorkers.append(workername)

    def maybeStartBuildsForAllBuilders(self):
        if False:
            return 10
        self.buildsStartedForWorkers += self.builders.keys()

    def workerLost(self, bot):
        if False:
            i = 10
            return i + 15
        pass

    def cleanShutdown(self, quickMode=False, stopReactor=True):
        if False:
            for i in range(10):
                print('nop')
        self.shuttingDown = True
        if self.delayShutdown:
            self.shutdownDeferred = defer.Deferred()
            return self.shutdownDeferred
        return None