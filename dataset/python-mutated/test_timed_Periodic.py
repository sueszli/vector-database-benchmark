from twisted.internet import defer
from twisted.trial import unittest
from buildbot import config
from buildbot.schedulers import timed
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler

class TestException(Exception):
    pass

class Periodic(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):
    OBJECTID = 23
    SCHEDULERID = 3

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        self.setUpScheduler()

    def makeScheduler(self, firstBuildDuration=0, firstBuildError=False, exp_branch=None, **kwargs):
        if False:
            while True:
                i = 10
        self.sched = sched = timed.Periodic(**kwargs)
        sched._reactor = self.reactor
        self.attachScheduler(self.sched, self.OBJECTID, self.SCHEDULERID)
        self.events = []

        def addBuildsetForSourceStampsWithDefaults(reason, sourcestamps, waited_for=False, properties=None, builderNames=None, **kw):
            if False:
                print('Hello World!')
            self.assertIn('Periodic scheduler named', reason)
            isFirst = not self.events
            if self.reactor.seconds() == 0 and firstBuildError:
                raise TestException()
            self.events.append(f'B@{int(self.reactor.seconds())}')
            if isFirst and firstBuildDuration:
                d = defer.Deferred()
                self.reactor.callLater(firstBuildDuration, d.callback, None)
                return d
            return defer.succeed(None)
        sched.addBuildsetForSourceStampsWithDefaults = addBuildsetForSourceStampsWithDefaults
        self.state = {}

        def getState(k, default):
            if False:
                while True:
                    i = 10
            return defer.succeed(self.state.get(k, default))
        sched.getState = getState

        def setState(k, v):
            if False:
                for i in range(10):
                    print('nop')
            self.state[k] = v
            return defer.succeed(None)
        sched.setState = setState
        return sched

    def test_constructor_invalid(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(config.ConfigErrors):
            timed.Periodic(name='test', builderNames=['test'], periodicBuildTimer=-2)

    def test_constructor_no_reason(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=10)
        self.assertEqual(sched.reason, "The Periodic scheduler named 'test' triggered this build")

    def test_constructor_reason(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=10, reason='periodic')
        self.assertEqual(sched.reason, 'periodic')

    def test_iterations_simple(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        sched.activate()
        self.reactor.advance(0)
        while self.reactor.seconds() < 30:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@0', 'B@13', 'B@26'])
        self.assertEqual(self.state.get('last_build'), 26)
        d = sched.deactivate()
        return d

    def test_iterations_simple_branch(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(exp_branch='newfeature', name='test', builderNames=['test'], periodicBuildTimer=13, branch='newfeature')
        sched.activate()
        self.reactor.advance(0)
        while self.reactor.seconds() < 30:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@0', 'B@13', 'B@26'])
        self.assertEqual(self.state.get('last_build'), 26)
        d = sched.deactivate()
        return d

    def test_iterations_long(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=10, firstBuildDuration=15)
        sched.activate()
        self.reactor.advance(0)
        while self.reactor.seconds() < 40:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@0', 'B@15', 'B@25', 'B@35'])
        self.assertEqual(self.state.get('last_build'), 35)
        d = sched.deactivate()
        return d

    @defer.inlineCallbacks
    def test_start_build_error(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=10, firstBuildError=True)
        yield sched.activate()
        self.reactor.advance(0)
        while self.reactor.seconds() < 40:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@10', 'B@20', 'B@30', 'B@40'])
        self.assertEqual(self.state.get('last_build'), 40)
        self.assertEqual(1, len(self.flushLoggedErrors(TestException)))
        yield sched.deactivate()

    def test_iterations_stop_while_starting_build(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13, firstBuildDuration=6)
        sched.activate()
        self.reactor.advance(0)
        self.reactor.advance(3)
        d = sched.deactivate()
        d.addCallback(lambda _: self.events.append(f'STOP@{int(self.reactor.seconds())}'))
        while self.reactor.seconds() < 40:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@0', 'STOP@6'])
        self.assertEqual(self.state.get('last_build'), 0)
        return d

    def test_iterations_with_initial_state(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        self.state['last_build'] = self.reactor.seconds() - 7
        sched.activate()
        self.reactor.advance(0)
        while self.reactor.seconds() < 30:
            self.reactor.advance(1)
        self.assertEqual(self.events, ['B@6', 'B@19'])
        self.assertEqual(self.state.get('last_build'), 19)
        d = sched.deactivate()
        return d

    @defer.inlineCallbacks
    def test_getNextBuildTime_None(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        t = (yield sched.getNextBuildTime(None))
        self.assertEqual(t, 0)

    @defer.inlineCallbacks
    def test_getNextBuildTime_given(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        t = (yield sched.getNextBuildTime(20))
        self.assertEqual(t, 33)

    @defer.inlineCallbacks
    def test_enabled_callback(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        expectedValue = not sched.enabled
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, expectedValue)
        expectedValue = not sched.enabled
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, expectedValue)

    @defer.inlineCallbacks
    def test_disabled_activate(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, False)
        r = (yield sched.activate())
        self.assertEqual(r, None)

    @defer.inlineCallbacks
    def test_disabled_deactivate(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, False)
        r = (yield sched.deactivate())
        self.assertEqual(r, None)

    @defer.inlineCallbacks
    def test_disabled_start_build(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], periodicBuildTimer=13)
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, False)
        r = (yield sched.startBuild())
        self.assertEqual(r, None)