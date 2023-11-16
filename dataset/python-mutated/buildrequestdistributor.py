import copy
import math
import random
from datetime import datetime
from twisted.internet import defer
from twisted.python import log
from twisted.python.failure import Failure
from buildbot.data import resultspec
from buildbot.process import metrics
from buildbot.process.buildrequest import BuildRequest
from buildbot.util import deferwaiter
from buildbot.util import epoch2datetime
from buildbot.util import service
from buildbot.util.async_sort import async_sort

class BuildChooserBase:

    def __init__(self, bldr, master):
        if False:
            while True:
                i = 10
        self.bldr = bldr
        self.master = master
        self.breqCache = {}
        self.unclaimedBrdicts = None

    @defer.inlineCallbacks
    def chooseNextBuild(self):
        if False:
            return 10
        (worker, breq) = (yield self.popNextBuild())
        if not worker or not breq:
            return (None, None)
        return (worker, [breq])

    def popNextBuild(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Subclasses must implement this!')

    @defer.inlineCallbacks
    def _fetchUnclaimedBrdicts(self):
        if False:
            i = 10
            return i + 15
        if self.unclaimedBrdicts is None:
            brdicts = (yield self.master.data.get(('builders', (yield self.bldr.getBuilderId()), 'buildrequests'), [resultspec.Filter('claimed', 'eq', [False])]))
            brdicts.sort(key=lambda brd: brd['buildrequestid'])
            self.unclaimedBrdicts = brdicts
        return self.unclaimedBrdicts

    @defer.inlineCallbacks
    def _getBuildRequestForBrdict(self, brdict):
        if False:
            i = 10
            return i + 15
        breq = self.breqCache.get(brdict['buildrequestid'])
        if not breq:
            breq = (yield BuildRequest.fromBrdict(self.master, brdict))
            if breq:
                self.breqCache[brdict['buildrequestid']] = breq
        return breq

    def _getBrdictForBuildRequest(self, breq):
        if False:
            print('Hello World!')
        if breq is None:
            return None
        brid = breq.id
        for brdict in self.unclaimedBrdicts:
            if brid == brdict['buildrequestid']:
                return brdict
        return None

    def _removeBuildRequest(self, breq):
        if False:
            i = 10
            return i + 15
        if breq is None:
            return
        brdict = self._getBrdictForBuildRequest(breq)
        if brdict is not None:
            self.unclaimedBrdicts.remove(brdict)
        if breq.id in self.breqCache:
            del self.breqCache[breq.id]

    def _getUnclaimedBuildRequests(self):
        if False:
            return 10
        return defer.gatherResults([self._getBuildRequestForBrdict(brdict) for brdict in self.unclaimedBrdicts])

class BasicBuildChooser(BuildChooserBase):

    def __init__(self, bldr, master):
        if False:
            i = 10
            return i + 15
        super().__init__(bldr, master)
        self.nextWorker = self.bldr.config.nextWorker
        if not self.nextWorker:
            self.nextWorker = lambda _, workers, __: random.choice(workers) if workers else None
        self.workerpool = self.bldr.getAvailableWorkers()
        self.preferredWorkers = []
        self.nextBuild = self.bldr.config.nextBuild

    @defer.inlineCallbacks
    def popNextBuild(self):
        if False:
            return 10
        nextBuild = (None, None)
        while True:
            breq = (yield self._getNextUnclaimedBuildRequest())
            if not breq:
                break
            worker = (yield self._popNextWorker(breq))
            if not worker:
                break
            self._removeBuildRequest(breq)
            recycledWorkers = []
            while worker:
                canStart = (yield self.canStartBuild(worker, breq))
                if canStart:
                    break
                recycledWorkers.append(worker)
                worker = (yield self._popNextWorker(breq))
            if recycledWorkers:
                self._unpopWorkers(recycledWorkers)
            if worker:
                nextBuild = (worker, breq)
                break
        return nextBuild

    @defer.inlineCallbacks
    def _getNextUnclaimedBuildRequest(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._fetchUnclaimedBrdicts()
        if not self.unclaimedBrdicts:
            return None
        if self.nextBuild:
            breqs = (yield self._getUnclaimedBuildRequests())
            try:
                nextBreq = (yield self.nextBuild(self.bldr, breqs))
                if nextBreq not in breqs:
                    nextBreq = None
            except Exception:
                log.err(Failure(), f"from _getNextUnclaimedBuildRequest for builder '{self.bldr}'")
                nextBreq = None
        else:
            brdict = sorted(self.unclaimedBrdicts.data, key=lambda b: b['priority'], reverse=True)[0]
            nextBreq = (yield self._getBuildRequestForBrdict(brdict))
        return nextBreq

    @defer.inlineCallbacks
    def _popNextWorker(self, buildrequest):
        if False:
            return 10
        if self.preferredWorkers:
            worker = self.preferredWorkers.pop(0)
            return worker
        while self.workerpool:
            try:
                worker = (yield self.nextWorker(self.bldr, self.workerpool, buildrequest))
            except Exception:
                log.err(Failure(), f"from nextWorker for builder '{self.bldr}'")
                worker = None
            if not worker or worker not in self.workerpool:
                break
            self.workerpool.remove(worker)
            return worker
        return None

    def _unpopWorkers(self, workers):
        if False:
            while True:
                i = 10
        self.preferredWorkers[:0] = workers

    def canStartBuild(self, worker, breq):
        if False:
            i = 10
            return i + 15
        return self.bldr.canStartBuild(worker, breq)

class BuildRequestDistributor(service.AsyncMultiService):
    """
    Special-purpose class to handle distributing build requests to builders by
    calling their C{maybeStartBuild} method.

    This takes account of the C{prioritizeBuilders} configuration, and is
    highly re-entrant; that is, if a new build request arrives while builders
    are still working on the previous build request, then this class will
    correctly re-prioritize invocations of builders' C{maybeStartBuild}
    methods.
    """
    BuildChooser = BasicBuildChooser

    def __init__(self, botmaster):
        if False:
            print('Hello World!')
        super().__init__()
        self.botmaster = botmaster
        self.pending_builders_lock = defer.DeferredLock()
        self._pending_builders = []
        self.activity_lock = defer.DeferredLock()
        self.active = False
        self._deferwaiter = deferwaiter.DeferWaiter()
        self._activity_loop_deferred = None

    @defer.inlineCallbacks
    def stopService(self):
        if False:
            print('Hello World!')
        yield self.activity_lock.run(service.AsyncService.stopService, self)
        yield self._deferwaiter.wait()

    @defer.inlineCallbacks
    def maybeStartBuildsOn(self, new_builders):
        if False:
            i = 10
            return i + 15
        '\n        Try to start any builds that can be started right now.  This function\n        returns immediately, and promises to trigger those builders\n        eventually.\n\n        @param new_builders: names of new builders that should be given the\n        opportunity to check for new requests.\n        '
        if not self.running:
            return
        try:
            yield self._deferwaiter.add(self._maybeStartBuildsOn(new_builders))
        except Exception as e:
            log.err(e, f'while starting builds on {new_builders}')

    @defer.inlineCallbacks
    def _maybeStartBuildsOn(self, new_builders):
        if False:
            return 10
        new_builders = set(new_builders)
        existing_pending = set(self._pending_builders)
        if new_builders < existing_pending:
            return None

        @defer.inlineCallbacks
        def resetPendingBuildersList(new_builders):
            if False:
                for i in range(10):
                    print('nop')
            try:
                existing_pending = set(self._pending_builders)
                self._pending_builders = (yield self._sortBuilders(list(existing_pending | new_builders)))
                if not self.active:
                    self._activity_loop_deferred = self._activityLoop()
            except Exception:
                log.err(Failure(), f'while attempting to start builds on {self.name}')
        yield self.pending_builders_lock.run(resetPendingBuildersList, new_builders)
        return None

    @defer.inlineCallbacks
    def _defaultSorter(self, master, builders):
        if False:
            print('Hello World!')
        timer = metrics.Timer('BuildRequestDistributor._defaultSorter()')
        timer.start()

        @defer.inlineCallbacks
        def key(bldr):
            if False:
                for i in range(10):
                    print('nop')
            priority = (yield bldr.get_highest_priority())
            if priority is None:
                priority = -math.inf
            time = (yield bldr.getOldestRequestTime())
            if time is None:
                time = math.inf
            elif isinstance(time, datetime):
                time = time.timestamp()
            return (-priority, time, bldr.name)
        yield async_sort(builders, key)
        timer.stop()
        return builders

    @defer.inlineCallbacks
    def _sortBuilders(self, buildernames):
        if False:
            while True:
                i = 10
        timer = metrics.Timer('BuildRequestDistributor._sortBuilders()')
        timer.start()
        builders_dict = self.botmaster.builders
        builders = [builders_dict.get(n) for n in buildernames if n in builders_dict]
        sorter = self.master.config.prioritizeBuilders
        if not sorter:
            sorter = self._defaultSorter
        try:
            builders = (yield sorter(self.master, builders))
        except Exception:
            log.err(Failure(), 'prioritizing builders; order unspecified')
        rv = [b.name for b in builders]
        timer.stop()
        return rv

    @defer.inlineCallbacks
    def _activityLoop(self):
        if False:
            i = 10
            return i + 15
        self.active = True
        timer = metrics.Timer('BuildRequestDistributor._activityLoop()')
        timer.start()
        pending_builders = []
        while True:
            yield self.activity_lock.acquire()
            if not self.running:
                self.activity_lock.release()
                break
            if not pending_builders:
                yield self.pending_builders_lock.acquire()
                if not self._pending_builders:
                    self.pending_builders_lock.release()
                    self.activity_lock.release()
                    break
                pending_builders = copy.copy(self._pending_builders)
                self._pending_builders = []
                self.pending_builders_lock.release()
            bldr_name = pending_builders.pop(0)
            bldr = self.botmaster.builders.get(bldr_name)
            try:
                if bldr:
                    yield self._maybeStartBuildsOnBuilder(bldr)
            except Exception:
                log.err(Failure(), f"from maybeStartBuild for builder '{bldr_name}'")
            self.activity_lock.release()
        timer.stop()
        self.active = False

    @defer.inlineCallbacks
    def _maybeStartBuildsOnBuilder(self, bldr):
        if False:
            print('Hello World!')
        bc = self.createBuildChooser(bldr, self.master)
        while True:
            (worker, breqs) = (yield bc.chooseNextBuild())
            if not worker or not breqs:
                break
            brids = [br.id for br in breqs]
            claimed_at_epoch = self.master.reactor.seconds()
            claimed_at = epoch2datetime(claimed_at_epoch)
            if not (yield self.master.data.updates.claimBuildRequests(brids, claimed_at=claimed_at)):
                bc = self.createBuildChooser(bldr, self.master)
                continue
            buildStarted = (yield bldr.maybeStartBuild(worker, breqs))
            if not buildStarted:
                yield self.master.data.updates.unclaimBuildRequests(brids)
                self.botmaster.maybeStartBuildsForBuilder(self.name)

    def createBuildChooser(self, bldr, master):
        if False:
            return 10
        return self.BuildChooser(bldr, master)

    @defer.inlineCallbacks
    def _waitForFinish(self):
        if False:
            i = 10
            return i + 15
        if self._activity_loop_deferred is not None:
            yield self._activity_loop_deferred