from twisted.internet import task
from twisted.trial import unittest
from buildbot.process import properties
from buildbot.schedulers import timed
from buildbot.test import fakedb
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler

class NightlyTriggerable(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):
    SCHEDULERID = 327
    OBJECTID = 1327

    def makeScheduler(self, firstBuildDuration=0, **kwargs):
        if False:
            i = 10
            return i + 15
        sched = self.attachScheduler(timed.NightlyTriggerable(**kwargs), self.OBJECTID, self.SCHEDULERID, overrideBuildsetMethods=True, createBuilderDB=True)
        self.clock = sched._reactor = task.Clock()
        return sched

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.setUpScheduler()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tearDownScheduler()

    def assertBuildsetAdded(self, sourcestamps=None, properties=None):
        if False:
            for i in range(10):
                print('nop')
        if sourcestamps is None:
            sourcestamps = {}
        if properties is None:
            properties = {}
        properties['scheduler'] = ('test', 'Scheduler')
        self.assertEqual(self.addBuildsetCalls, [('addBuildsetForSourceStampsWithDefaults', {'builderNames': None, 'priority': None, 'properties': properties, 'reason': "The NightlyTriggerable scheduler named 'test' triggered this build", 'sourcestamps': sourcestamps, 'waited_for': False})])
        self.addBuildsetCalls = []

    def assertNoBuildsetAdded(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.addBuildsetCalls, [])

    def test_constructor_no_reason(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'])
        self.assertEqual(sched.reason, "The NightlyTriggerable scheduler named 'test' triggered this build")

    def test_constructor_reason(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], reason='hourlytriggerable')
        self.assertEqual(sched.reason, 'hourlytriggerable')

    def test_constructor_month(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], month='1')
        self.assertEqual(sched.month, '1')

    def test_timer_noBuilds(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5])
        sched.activate()
        self.clock.advance(60 * 60)
        self.assertEqual(self.addBuildsetCalls, [])

    def test_timer_oneTrigger(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        sched.activate()
        sched.trigger(False, [{'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb'}], set_props=None)
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])

    def test_timer_twoTriggers(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        sched.activate()
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev1', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev2', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev2'}])

    def test_timer_oneTrigger_then_noBuild(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        sched.activate()
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])
        self.clock.advance(60 * 60)
        self.assertNoBuildsetAdded()

    def test_timer_oneTriggers_then_oneTrigger(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        sched.activate()
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev1', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev1'}])
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev2', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev2'}])

    def test_savedTrigger(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        value_json = '[ [ {"codebase": "cb", "project": "p", "repository": "r", "branch": "br", "revision": "myrev"} ], {}, null, null ]'
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable'), fakedb.ObjectState(objectid=self.SCHEDULERID, name='lastTrigger', value_json=value_json)])
        sched.activate()
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])

    def test_savedTrigger_dict(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        value_json = '[ { "cb": {"codebase": "cb", "project": "p", "repository": "r", "branch": "br", "revision": "myrev"} }, {}, null, null ]'
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable'), fakedb.ObjectState(objectid=self.SCHEDULERID, name='lastTrigger', value_json=value_json)])
        sched.activate()
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])

    def test_saveTrigger(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable')])
        sched.activate()
        (_, d) = sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)

        @d.addCallback
        def cb(_):
            if False:
                while True:
                    i = 10
            self.db.state.assertState(self.SCHEDULERID, lastTrigger=[[{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], {}, None, None])
        return d

    def test_saveTrigger_noTrigger(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable')])
        sched.activate()
        (_, d) = sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], set_props=None)
        self.clock.advance(60 * 60)

        @d.addCallback
        def cb(_):
            if False:
                for i in range(10):
                    print('nop')
            self.db.state.assertState(self.SCHEDULERID, lastTrigger=None)
        return d

    def test_triggerProperties(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable')])
        sched.activate()
        sched.trigger(False, [{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], properties.Properties(testprop='test'))
        self.db.state.assertState(self.SCHEDULERID, lastTrigger=[[{'codebase': 'cb', 'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r'}], {'testprop': ['test', 'TEST']}, None, None])
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(properties={'testprop': ('test', 'TEST')}, sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])

    def test_savedProperties(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[5], codebases={'cb': {'repository': 'annoying'}})
        value_json = '[ [ {"codebase": "cb", "project": "p", "repository": "r", "branch": "br", "revision": "myrev"} ], {"testprop": ["test", "TEST"]}, null, null ]'
        self.db.insert_test_data([fakedb.Object(id=self.SCHEDULERID, name='test', class_name='NightlyTriggerable'), fakedb.ObjectState(objectid=self.SCHEDULERID, name='lastTrigger', value_json=value_json)])
        sched.activate()
        self.clock.advance(60 * 60)
        self.assertBuildsetAdded(properties={'testprop': ('test', 'TEST')}, sourcestamps=[{'codebase': 'cb', 'branch': 'br', 'project': 'p', 'repository': 'r', 'revision': 'myrev'}])