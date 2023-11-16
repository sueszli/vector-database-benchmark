import random
from unittest import mock
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
from buildbot import config
from buildbot.db import buildrequests
from buildbot.process import buildrequestdistributor
from buildbot.process import factory
from buildbot.test import fakedb
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.warnings import assertProducesWarning
from buildbot.util import epoch2datetime
from buildbot.util.eventual import fireEventually
from buildbot.warnings import DeprecatedApiWarning

def nth_worker(n):
    if False:
        print('Hello World!')

    def pick_nth_by_name(builder, workers=None, br=None):
        if False:
            i = 10
            return i + 15
        if workers is None:
            workers = builder
        workers = workers[:]
        workers.sort(key=lambda a: a.name)
        return workers[n]
    return pick_nth_by_name

class TestBRDBase(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.botmaster = mock.Mock(name='botmaster')
        self.botmaster.builders = {}
        self.builders = {}

        def prioritizeBuilders(master, builders):
            if False:
                i = 10
                return i + 15
            return sorted(builders, key=lambda b1: b1.name)
        self.master = self.botmaster.master = fakemaster.make_master(self, wantData=True, wantDb=True)
        self.master.caches = fakemaster.FakeCaches()
        self.master.config.prioritizeBuilders = prioritizeBuilders
        self.brd = buildrequestdistributor.BuildRequestDistributor(self.botmaster)
        self.brd.parent = self.botmaster
        self.brd.startService()
        self.base_rows = [fakedb.SourceStamp(id=21), fakedb.Builder(id=77, name='A'), fakedb.Buildset(id=11, reason='because'), fakedb.BuildsetSourceStamp(sourcestampid=21, buildsetid=11)]

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.brd.running:
            return self.brd.stopService()
        return None

    def make_workers(self, worker_count):
        if False:
            i = 10
            return i + 15
        rows = self.base_rows[:]
        for i in range(worker_count):
            self.addWorkers({f'test-worker{i}': 1})
            rows.append(fakedb.Buildset(id=100 + i, reason='because'))
            rows.append(fakedb.BuildsetSourceStamp(buildsetid=100 + i, sourcestampid=21))
            rows.append(fakedb.BuildRequest(id=10 + i, buildsetid=100 + i, builderid=77))
        return rows

    def addWorkers(self, workerforbuilders):
        if False:
            print('Hello World!')
        'C{workerforbuilders} maps name : available'
        for (name, avail) in workerforbuilders.items():
            wfb = mock.Mock(spec=['isAvailable'], name=name)
            wfb.name = name
            wfb.isAvailable.return_value = avail
            for bldr in self.builders.values():
                bldr.workers.append(wfb)

    @defer.inlineCallbacks
    def createBuilder(self, name, builderid=None, builder_config=None):
        if False:
            while True:
                i = 10
        if builderid is None:
            b = fakedb.Builder(name=name)
            yield self.master.db.insert_test_data([b])
            builderid = b.id
        bldr = mock.Mock(name=name)
        bldr.name = name
        self.botmaster.builders[name] = bldr
        self.builders[name] = bldr

        def maybeStartBuild(worker, builds):
            if False:
                i = 10
                return i + 15
            self.startedBuilds.append((worker.name, builds))
            d = defer.Deferred()
            self.reactor.callLater(0, d.callback, True)
            return d
        bldr.maybeStartBuild = maybeStartBuild
        bldr.getCollapseRequestsFn = lambda : False
        bldr.workers = []
        bldr.getAvailableWorkers = lambda : [w for w in bldr.workers if w.isAvailable()]
        bldr.getBuilderId = lambda : builderid
        if builder_config is None:
            bldr.config.nextWorker = None
            bldr.config.nextBuild = None
        else:
            bldr.config = builder_config

        def canStartBuild(*args):
            if False:
                for i in range(10):
                    print('nop')
            can = bldr.config.canStartBuild
            return not can or can(*args)
        bldr.canStartBuild = canStartBuild
        return bldr

    @defer.inlineCallbacks
    def addBuilders(self, names):
        if False:
            for i in range(10):
                print('nop')
        self.startedBuilds = []
        for name in names:
            yield self.createBuilder(name)

    def assertMyClaims(self, brids):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.master.data.updates.claimedBuildRequests, set(brids))

class Test(TestBRDBase):

    def checkAllCleanedUp(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.brd.pending_builders_lock.locked, False)
        self.assertEqual(self.brd.activity_lock.locked, False)
        self.assertEqual(self.brd.active, False)

    def useMock_maybeStartBuildsOnBuilder(self):
        if False:
            print('Hello World!')
        self.maybeStartBuildsOnBuilder_calls = []

        def maybeStartBuildsOnBuilder(bldr):
            if False:
                return 10
            self.assertIdentical(self.builders[bldr.name], bldr)
            self.maybeStartBuildsOnBuilder_calls.append(bldr.name)
            return fireEventually()
        self.brd._maybeStartBuildsOnBuilder = maybeStartBuildsOnBuilder

    def removeBuilder(self, name):
        if False:
            for i in range(10):
                print('nop')
        del self.builders[name]
        del self.botmaster.builders[name]

    @defer.inlineCallbacks
    def test_maybeStartBuildsOn_simple(self):
        if False:
            while True:
                i = 10
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(['bldr1'])
        yield self.brd.maybeStartBuildsOn(['bldr1'])
        yield self.brd._waitForFinish()
        self.assertEqual(self.maybeStartBuildsOnBuilder_calls, ['bldr1'])
        self.checkAllCleanedUp()

    @defer.inlineCallbacks
    def test_maybeStartBuildsOn_parallel(self):
        if False:
            i = 10
            return i + 15
        builders = [f'bldr{i:02}' for i in range(15)]

        def slow_sorter(master, bldrs):
            if False:
                return 10
            bldrs.sort(key=lambda b1: b1.name)
            d = defer.Deferred()
            self.reactor.callLater(0, d.callback, bldrs)

            def done(_):
                if False:
                    i = 10
                    return i + 15
                return _
            d.addCallback(done)
            return d
        self.master.config.prioritizeBuilders = slow_sorter
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(builders)
        for bldr in builders:
            yield self.brd.maybeStartBuildsOn([bldr])
        yield self.brd._waitForFinish()
        self.assertEqual(self.maybeStartBuildsOnBuilder_calls, builders)
        self.checkAllCleanedUp()

    @defer.inlineCallbacks
    def test_maybeStartBuildsOn_exception(self):
        if False:
            while True:
                i = 10
        self.addBuilders(['bldr1'])

        def _maybeStartBuildsOnBuilder(n):
            if False:
                for i in range(10):
                    print('nop')
            d = defer.Deferred()
            self.reactor.callLater(0, d.errback, failure.Failure(RuntimeError('oh noes')))
            return d
        self.brd._maybeStartBuildsOnBuilder = _maybeStartBuildsOnBuilder
        yield self.brd.maybeStartBuildsOn(['bldr1'])
        yield self.brd._waitForFinish()
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
        self.checkAllCleanedUp()

    @defer.inlineCallbacks
    def test_maybeStartBuildsOn_collapsing(self):
        if False:
            print('Hello World!')
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(['bldr1', 'bldr2', 'bldr3'])
        yield self.brd.maybeStartBuildsOn(['bldr3'])
        yield self.brd.maybeStartBuildsOn(['bldr2', 'bldr1'])
        yield self.brd.maybeStartBuildsOn(['bldr4'])
        yield self.brd.maybeStartBuildsOn(['bldr2'])
        yield self.brd.maybeStartBuildsOn(['bldr3', 'bldr2'])
        yield self.brd._waitForFinish()
        self.assertEqual(self.maybeStartBuildsOnBuilder_calls, ['bldr3', 'bldr1', 'bldr2', 'bldr3'])
        self.checkAllCleanedUp()

    @defer.inlineCallbacks
    def test_maybeStartBuildsOn_builders_missing(self):
        if False:
            print('Hello World!')
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(['bldr1', 'bldr2', 'bldr3'])
        yield self.brd.maybeStartBuildsOn(['bldr1', 'bldr2', 'bldr3'])
        self.removeBuilder('bldr2')
        self.removeBuilder('bldr3')
        yield self.brd._waitForFinish()
        self.assertEqual(self.maybeStartBuildsOnBuilder_calls, ['bldr1'])
        self.checkAllCleanedUp()

    @defer.inlineCallbacks
    def do_test_sortBuilders(self, prioritizeBuilders, oldestRequestTimes, highestPriorities, expected, returnDeferred=False):
        if False:
            while True:
                i = 10
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(list(oldestRequestTimes))
        self.master.config.prioritizeBuilders = prioritizeBuilders

        def mklambda(t):
            if False:
                return 10
            if returnDeferred:
                return lambda : defer.succeed(t)
            return lambda : t
        for (n, t) in oldestRequestTimes.items():
            if t is not None:
                t = epoch2datetime(t)
            self.builders[n].getOldestRequestTime = mklambda(t)
        for (n, t) in highestPriorities.items():
            self.builders[n].get_highest_priority = mklambda(t)
        result = (yield self.brd._sortBuilders(list(oldestRequestTimes)))
        self.assertEqual(result, expected)
        self.checkAllCleanedUp()

    def test_sortBuilders_default_sync(self):
        if False:
            return 10
        return self.do_test_sortBuilders(None, {'bldr1': 777, 'bldr2': 999, 'bldr3': 888}, {'bldr1': 10, 'bldr2': 15, 'bldr3': 5}, ['bldr2', 'bldr1', 'bldr3'])

    def test_sortBuilders_default_asyn(self):
        if False:
            while True:
                i = 10
        return self.do_test_sortBuilders(None, {'bldr1': 777, 'bldr2': 999, 'bldr3': 888}, {'bldr1': 10, 'bldr2': 15, 'bldr3': 5}, ['bldr2', 'bldr1', 'bldr3'], returnDeferred=True)

    def test_sortBuilders_default_None(self):
        if False:
            print('Hello World!')
        return self.do_test_sortBuilders(None, {'bldr1': 777, 'bldr2': None, 'bldr3': 888}, {'bldr1': 10, 'bldr2': None, 'bldr3': 5}, ['bldr1', 'bldr3', 'bldr2'])

    def test_sortBuilders_default_priority_match(self):
        if False:
            print('Hello World!')
        return self.do_test_sortBuilders(None, {'bldr1': 777, 'bldr2': 999, 'bldr3': 888}, {'bldr1': 10, 'bldr2': 10, 'bldr3': 10}, ['bldr1', 'bldr3', 'bldr2'])

    def test_sortBuilders_custom(self):
        if False:
            return 10

        def prioritizeBuilders(master, builders):
            if False:
                print('Hello World!')
            self.assertIdentical(master, self.master)
            return sorted(builders, key=lambda b: b.name)
        return self.do_test_sortBuilders(prioritizeBuilders, {'bldr1': 1, 'bldr2': 1, 'bldr3': 1}, {'bldr1': 10, 'bldr2': 15, 'bldr3': 5}, ['bldr1', 'bldr2', 'bldr3'])

    def test_sortBuilders_custom_async(self):
        if False:
            while True:
                i = 10

        def prioritizeBuilders(master, builders):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIdentical(master, self.master)
            return defer.succeed(sorted(builders, key=lambda b: b.name))
        return self.do_test_sortBuilders(prioritizeBuilders, {'bldr1': 1, 'bldr2': 1, 'bldr3': 1}, {'bldr1': 10, 'bldr2': 15, 'bldr3': 5}, ['bldr1', 'bldr2', 'bldr3'])

    @defer.inlineCallbacks
    def test_sortBuilders_custom_exception(self):
        if False:
            i = 10
            return i + 15
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(['x', 'y'])

        def fail(m, b):
            if False:
                while True:
                    i = 10
            raise RuntimeError('oh noes')
        self.master.config.prioritizeBuilders = fail
        result = (yield self.brd._sortBuilders(['y', 'x']))
        self.assertEqual(result, ['y', 'x'])
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    @defer.inlineCallbacks
    def test_stopService(self):
        if False:
            return 10
        self.useMock_maybeStartBuildsOnBuilder()
        self.addBuilders(['A', 'B'])
        oldMSBOB = self.brd._maybeStartBuildsOnBuilder

        def maybeStartBuildsOnBuilder(bldr):
            if False:
                return 10
            d = oldMSBOB(bldr)
            stop_d = self.brd.stopService()
            stop_d.addCallback(lambda _: self.maybeStartBuildsOnBuilder_calls.append('(stopped)'))
            d.addCallback(lambda _: self.maybeStartBuildsOnBuilder_calls.append('finished'))
            return d
        self.brd._maybeStartBuildsOnBuilder = maybeStartBuildsOnBuilder
        yield self.brd.maybeStartBuildsOn(['A', 'B'])
        yield self.brd._waitForFinish()
        self.assertEqual(self.maybeStartBuildsOnBuilder_calls, ['A', 'finished', '(stopped)'])

class TestMaybeStartBuilds(TestBRDBase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        yield super().setUp()
        self.startedBuilds = []
        self.bldr = (yield self.createBuilder('A', builderid=77))
        self.builders['A'] = self.bldr

    def assertBuildsStarted(self, exp):
        if False:
            return 10
        builds_started = [(worker, [br.id for br in breqs]) for (worker, breqs) in self.startedBuilds]
        self.assertEqual(builds_started, exp)

    @defer.inlineCallbacks
    def do_test_maybeStartBuildsOnBuilder(self, rows=None, exp_claims=None, exp_builds=None):
        if False:
            for i in range(10):
                print('nop')
        rows = rows or []
        exp_claims = exp_claims or []
        exp_builds = exp_builds or []
        yield self.master.db.insert_test_data(rows)
        yield self.brd._maybeStartBuildsOnBuilder(self.bldr)
        self.assertMyClaims(exp_claims)
        self.assertBuildsStarted(exp_builds)

    @defer.inlineCallbacks
    def test_no_buildrequests(self):
        if False:
            while True:
                i = 10
        self.addWorkers({'test-worker11': 1})
        yield self.do_test_maybeStartBuildsOnBuilder(exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_no_workerforbuilders(self):
        if False:
            print('Hello World!')
        rows = [fakedb.Builder(id=78, name='bldr'), fakedb.BuildRequest(id=11, buildsetid=10, builderid=78)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_limited_by_workers(self):
        if False:
            return 10
        self.addWorkers({'test-worker1': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000), fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10], exp_builds=[('test-worker1', [10])])

    @defer.inlineCallbacks
    def test_sorted_by_submit_time(self):
        if False:
            print('Hello World!')
        self.addWorkers({'test-worker1': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10], exp_builds=[('test-worker1', [10])])

    @defer.inlineCallbacks
    def test_limited_by_available_workers(self):
        if False:
            print('Hello World!')
        self.addWorkers({'test-worker1': 0, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10], exp_builds=[('test-worker2', [10])])

    @defer.inlineCallbacks
    def test_slow_db(self):
        if False:
            while True:
                i = 10
        self.addWorkers({'test-worker1': 1})
        old_getBuildRequests = self.master.db.buildrequests.getBuildRequests

        def longGetBuildRequests(*args, **kwargs):
            if False:
                while True:
                    i = 10
            res_d = old_getBuildRequests(*args, **kwargs)
            long_d = defer.Deferred()
            long_d.addCallback(lambda _: res_d)
            self.reactor.callLater(0, long_d.callback, None)
            return long_d
        self.master.db.buildrequests.getBuildRequests = longGetBuildRequests
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10], exp_builds=[('test-worker1', [10])])

    @defer.inlineCallbacks
    def test_limited_by_canStartBuild(self):
        if False:
            for i in range(10):
                print('nop')
        "Set the 'canStartBuild' value in the config to something\n        that limits the possible options."
        self.bldr.config.nextWorker = nth_worker(-1)
        pairs_tested = []

        def _canStartBuild(worker, breq):
            if False:
                return 10
            result = (worker.name, breq.id)
            pairs_tested.append(result)
            allowed = [('test-worker1', 10), ('test-worker3', 11)]
            return result in allowed
        self.bldr.config.canStartBuild = _canStartBuild
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1, 'test-worker3': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000), fakedb.BuildRequest(id=12, buildsetid=11, builderid=77, submitted_at=140000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10, 11], exp_builds=[('test-worker1', [10]), ('test-worker3', [11])])
        self.assertEqual(pairs_tested, [('test-worker3', 10), ('test-worker2', 10), ('test-worker1', 10), ('test-worker3', 11), ('test-worker2', 12)])

    @defer.inlineCallbacks
    def test_limited_by_canStartBuild_deferreds(self):
        if False:
            i = 10
            return i + 15
        self.bldr.config.nextWorker = nth_worker(-1)
        pairs_tested = []

        def _canStartBuild(worker, breq):
            if False:
                return 10
            result = (worker.name, breq.id)
            pairs_tested.append(result)
            allowed = [('test-worker1', 10), ('test-worker3', 11)]
            return defer.succeed(result in allowed)
        self.bldr.config.canStartBuild = _canStartBuild
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1, 'test-worker3': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000), fakedb.BuildRequest(id=12, buildsetid=11, builderid=77, submitted_at=140000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10, 11], exp_builds=[('test-worker1', [10]), ('test-worker3', [11])])
        self.assertEqual(pairs_tested, [('test-worker3', 10), ('test-worker2', 10), ('test-worker1', 10), ('test-worker3', 11), ('test-worker2', 12)])

    @defer.inlineCallbacks
    def test_unlimited(self):
        if False:
            return 10
        self.bldr.config.nextWorker = nth_worker(-1)
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[10, 11], exp_builds=[('test-worker2', [10]), ('test-worker1', [11])])

    @defer.inlineCallbacks
    def test_bldr_maybeStartBuild_fails_always(self):
        if False:
            return 10
        self.bldr.config.nextWorker = nth_worker(-1)

        def maybeStartBuild(worker, builds):
            if False:
                while True:
                    i = 10
            self.startedBuilds.append((worker.name, builds))
            return defer.succeed(False)
        self.bldr.maybeStartBuild = maybeStartBuild
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[('test-worker2', [10]), ('test-worker1', [11])])

    @defer.inlineCallbacks
    def test_bldr_maybeStartBuild_fails_once(self):
        if False:
            return 10
        self.bldr.config.nextWorker = nth_worker(-1)
        start_build_results = [False, True, True]

        def maybeStartBuild(worker, builds):
            if False:
                print('Hello World!')
            self.startedBuilds.append((worker.name, builds))
            return defer.succeed(start_build_results.pop(0))
        self.bldr.maybeStartBuild = maybeStartBuild
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.master.db.insert_test_data(rows)
        yield self.brd._maybeStartBuildsOnBuilder(self.bldr)
        self.assertMyClaims([11])
        self.assertBuildsStarted([('test-worker2', [10]), ('test-worker1', [11])])
        yield self.brd._maybeStartBuildsOnBuilder(self.bldr)
        self.assertMyClaims([10, 11])
        self.assertBuildsStarted([('test-worker2', [10]), ('test-worker1', [11]), ('test-worker2', [10])])

    @defer.inlineCallbacks
    def test_limited_by_requests(self):
        if False:
            return 10
        self.bldr.config.nextWorker = nth_worker(1)
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[11], exp_builds=[('test-worker2', [11])])

    @defer.inlineCallbacks
    def test_nextWorker_None(self):
        if False:
            i = 10
            return i + 15
        self.bldr.config.nextWorker = lambda _1, _2, _3: defer.succeed(None)
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_nextWorker_bogus(self):
        if False:
            print('Hello World!')
        self.bldr.config.nextWorker = lambda _1, _2, _3: defer.succeed(mock.Mock())
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_nextBuild_None(self):
        if False:
            i = 10
            return i + 15
        self.bldr.config.nextBuild = lambda _1, _2: defer.succeed(None)
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_nextBuild_bogus(self):
        if False:
            i = 10
            return i + 15
        self.bldr.config.nextBuild = lambda _1, _2: mock.Mock()
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])

    @defer.inlineCallbacks
    def test_nextBuild_fails(self):
        if False:
            for i in range(10):
                print('nop')

        def nextBuildRaises(*args):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('xx')
        self.bldr.config.nextBuild = nextBuildRaises
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=11, buildsetid=11, builderid=77)]
        result = self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[], exp_builds=[])
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))
        yield result

    @defer.inlineCallbacks
    def test_claim_race(self):
        if False:
            return 10
        self.bldr.config.nextWorker = nth_worker(0)
        old_claimBuildRequests = self.master.db.buildrequests.claimBuildRequests

        def claimBuildRequests(brids, claimed_at=None):
            if False:
                while True:
                    i = 10
            self.master.db.buildrequests.claimBuildRequests = old_claimBuildRequests
            assert 10 in brids
            self.master.db.buildrequests.fakeClaimBuildRequest(10, 136000, masterid=9999)
            return defer.fail(buildrequests.AlreadyClaimedError())
        self.master.db.buildrequests.claimBuildRequests = claimBuildRequests
        self.addWorkers({'test-worker1': 1, 'test-worker2': 1})
        rows = self.base_rows + [fakedb.BuildRequest(id=10, buildsetid=11, builderid=77, submitted_at=130000), fakedb.BuildRequest(id=11, buildsetid=11, builderid=77, submitted_at=135000)]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=[11], exp_builds=[('test-worker1', [11])])

    @defer.inlineCallbacks
    def do_test_nextWorker(self, nextWorker, exp_choice=None, exp_warning=False):
        if False:
            print('Hello World!')

        def makeBuilderConfig():
            if False:
                i = 10
                return i + 15
            return config.BuilderConfig(name='bldrconf', workernames=['wk1', 'wk2'], builddir='bdir', factory=factory.BuildFactory(), nextWorker=nextWorker)
        if exp_warning:
            with assertProducesWarning(DeprecatedApiWarning, message_pattern='nextWorker now takes a 3rd argument'):
                builder_config = makeBuilderConfig()
        else:
            builder_config = makeBuilderConfig()
        self.bldr = (yield self.createBuilder('B', builderid=78, builder_config=builder_config))
        for i in range(4):
            self.addWorkers({f'test-worker{i}': 1})
        rows = [fakedb.SourceStamp(id=21), fakedb.Builder(id=78, name='B'), fakedb.Buildset(id=12, reason='because'), fakedb.BuildsetSourceStamp(sourcestampid=21, buildsetid=12), fakedb.BuildRequest(id=12, buildsetid=12, builderid=78)]
        if exp_choice is None:
            exp_claims = []
            exp_builds = []
        else:
            exp_claims = [12]
            exp_builds = [(f'test-worker{exp_choice}', [12])]
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=exp_claims, exp_builds=exp_builds)

    def test_nextWorker_gets_buildrequest(self):
        if False:
            i = 10
            return i + 15

        def nextWorker(bldr, lst, br=None):
            if False:
                print('Hello World!')
            self.assertNotEqual(br, None)
        return self.do_test_nextWorker(nextWorker)

    def test_nextWorker_default(self):
        if False:
            return 10
        self.patch(random, 'choice', nth_worker(2))
        return self.do_test_nextWorker(None, exp_choice=2)

    def test_nextWorker_simple(self):
        if False:
            for i in range(10):
                print('nop')

        def nextWorker(bldr, lst, br=None):
            if False:
                print('Hello World!')
            self.assertIdentical(bldr, self.bldr)
            return lst[1]
        return self.do_test_nextWorker(nextWorker, exp_choice=1)

    def test_nextWorker_deferred(self):
        if False:
            for i in range(10):
                print('nop')

        def nextWorker(bldr, lst, br=None):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIdentical(bldr, self.bldr)
            return defer.succeed(lst[1])
        return self.do_test_nextWorker(nextWorker, exp_choice=1)

    @defer.inlineCallbacks
    def test_nextWorker_exception(self):
        if False:
            i = 10
            return i + 15

        def nextWorker(bldr, lst, br=None):
            if False:
                while True:
                    i = 10
            raise RuntimeError('')
        yield self.do_test_nextWorker(nextWorker)
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))

    @defer.inlineCallbacks
    def test_nextWorker_failure(self):
        if False:
            print('Hello World!')

        def nextWorker(bldr, lst, br=None):
            if False:
                for i in range(10):
                    print('nop')
            return defer.fail(failure.Failure(RuntimeError()))
        yield self.do_test_nextWorker(nextWorker)
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))

    @defer.inlineCallbacks
    def do_test_nextBuild(self, nextBuild, exp_choice=None):
        if False:
            while True:
                i = 10
        self.bldr.config.nextWorker = nth_worker(-1)
        self.bldr.config.nextBuild = nextBuild
        rows = self.make_workers(4)
        exp_claims = []
        exp_builds = []
        if exp_choice is not None:
            worker = 3
            for choice in exp_choice:
                exp_claims.append(choice)
                exp_builds.append((f'test-worker{worker}', [choice]))
                worker = worker - 1
        yield self.do_test_maybeStartBuildsOnBuilder(rows=rows, exp_claims=sorted(exp_claims), exp_builds=exp_builds)

    def test_nextBuild_default(self):
        if False:
            return 10
        'default chooses the first in the list, which should be the earliest'
        return self.do_test_nextBuild(None, exp_choice=[10, 11, 12, 13])

    def test_nextBuild_simple(self):
        if False:
            while True:
                i = 10

        def nextBuild(bldr, lst):
            if False:
                i = 10
                return i + 15
            self.assertIdentical(bldr, self.bldr)
            return lst[-1]
        return self.do_test_nextBuild(nextBuild, exp_choice=[13, 12, 11, 10])

    def test_nextBuild_deferred(self):
        if False:
            return 10

        def nextBuild(bldr, lst):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIdentical(bldr, self.bldr)
            return defer.succeed(lst[-1])
        return self.do_test_nextBuild(nextBuild, exp_choice=[13, 12, 11, 10])

    def test_nextBuild_exception(self):
        if False:
            for i in range(10):
                print('nop')

        def nextBuild(bldr, lst):
            if False:
                i = 10
                return i + 15
            raise RuntimeError('')
        result = self.do_test_nextBuild(nextBuild)
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))
        return result

    def test_nextBuild_failure(self):
        if False:
            while True:
                i = 10

        def nextBuild(bldr, lst):
            if False:
                print('Hello World!')
            return defer.fail(failure.Failure(RuntimeError()))
        result = self.do_test_nextBuild(nextBuild)
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))
        return result