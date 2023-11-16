from twisted.internet import defer
from twisted.python import log
from twisted.trial import unittest
from buildbot.process import properties
from buildbot.schedulers import triggerable
from buildbot.test import fakedb
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import interfaces
from buildbot.test.util import scheduler

class TriggerableInterfaceTest(unittest.TestCase, interfaces.InterfaceTests):

    def test_interface(self):
        if False:
            return 10
        self.assertInterfacesImplemented(triggerable.Triggerable)

class Triggerable(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):
    OBJECTID = 33
    SCHEDULERID = 13

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.reactor.advance(946684799)
        self.setUpScheduler()
        self.subscription = None

    def tearDown(self):
        if False:
            return 10
        self.tearDownScheduler()

    def makeScheduler(self, overrideBuildsetMethods=False, **kwargs):
        if False:
            return 10
        self.master.db.insert_test_data([fakedb.Builder(id=77, name='b')])
        sched = self.attachScheduler(triggerable.Triggerable(name='n', builderNames=['b'], **kwargs), self.OBJECTID, self.SCHEDULERID, overrideBuildsetMethods=overrideBuildsetMethods)
        return sched

    @defer.inlineCallbacks
    def assertTriggeredBuildset(self, idsDeferred, waited_for, properties=None, sourcestamps=None):
        if False:
            while True:
                i = 10
        if properties is None:
            properties = {}
        (bsid, brids) = (yield idsDeferred)
        properties.update({'scheduler': ('n', 'Scheduler')})
        self.assertEqual(self.master.db.buildsets.buildsets[bsid]['properties'], properties)
        buildset = (yield self.master.db.buildsets.getBuildset(bsid))
        from datetime import datetime
        from buildbot.util import UTC
        ssids = buildset.pop('sourcestamps')
        self.assertEqual(buildset, {'bsid': bsid, 'complete': False, 'complete_at': None, 'external_idstring': None, 'reason': "The Triggerable scheduler named 'n' triggered this build", 'results': -1, 'submitted_at': datetime(1999, 12, 31, 23, 59, 59, tzinfo=UTC), 'parent_buildid': None, 'parent_relationship': None})
        actual_sourcestamps = (yield defer.gatherResults([self.master.db.sourcestamps.getSourceStamp(ssid) for ssid in ssids]))
        self.assertEqual(len(sourcestamps), len(actual_sourcestamps))
        for (expected_ss, actual_ss) in zip(sourcestamps, actual_sourcestamps):
            actual_ss = actual_ss.copy()
            for key in list(actual_ss.keys()):
                if key not in expected_ss:
                    del actual_ss[key]
            self.assertEqual(expected_ss, actual_ss)
        for brid in brids.values():
            buildrequest = (yield self.master.db.buildrequests.getBuildRequest(brid))
            self.assertEqual(buildrequest, {'buildrequestid': brid, 'buildername': 'b', 'builderid': 77, 'buildsetid': bsid, 'claimed': False, 'claimed_at': None, 'complete': False, 'complete_at': None, 'claimed_by_masterid': None, 'priority': 0, 'results': -1, 'submitted_at': datetime(1999, 12, 31, 23, 59, 59, tzinfo=UTC), 'waited_for': waited_for})

    def sendCompletionMessage(self, bsid, results=3):
        if False:
            return 10
        self.master.mq.callConsumer(('buildsets', str(bsid), 'complete'), {'bsid': bsid, 'submitted_at': 100, 'complete': True, 'complete_at': 200, 'external_idstring': None, 'reason': 'triggering', 'results': results, 'sourcestamps': [], 'parent_buildid': None, 'parent_relationship': None})

    def test_constructor_no_reason(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler()
        self.assertEqual(sched.reason, None)

    def test_constructor_explicit_reason(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(reason='Because I said so')
        self.assertEqual(sched.reason, 'Because I said so')

    def test_constructor_priority_none(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(priority=None)
        self.assertEqual(sched.priority, None)

    def test_constructor_priority_int(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(priority=8)
        self.assertEqual(sched.priority, 8)

    def test_constructor_priority_function(self):
        if False:
            print('Hello World!')

        def sched_priority(builderNames, changesByCodebase):
            if False:
                print('Hello World!')
            return 0
        sched = self.makeScheduler(priority=sched_priority)
        self.assertEqual(sched.priority, sched_priority)

    def test_trigger(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(codebases={'cb': {'repository': 'r'}})
        self.assertEqual(sched.master.mq.qrefs, [])
        waited_for = True
        set_props = properties.Properties()
        set_props.setProperty('pr', 'op', 'test')
        ss = {'revision': 'myrev', 'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb'}
        (idsDeferred, d) = sched.trigger(waited_for, sourcestamps=[ss], set_props=set_props)
        self.reactor.advance(0)
        self.assertTriggeredBuildset(idsDeferred, waited_for, properties={'pr': ('op', 'test')}, sourcestamps=[{'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb', 'revision': 'myrev'}])
        self.fired = False

        @d.addCallback
        def fired(xxx_todo_changeme):
            if False:
                return 10
            (result, brids) = xxx_todo_changeme
            self.assertEqual(result, 3)
            self.assertEqual(brids, {77: 1000})
            self.fired = True
        d.addErrback(log.err)
        self.assertEqual([q.filter for q in sched.master.mq.qrefs], [('buildsets', None, 'complete')])
        self.assertFalse(self.fired)
        self.sendCompletionMessage(27)
        self.assertEqual([q.filter for q in sched.master.mq.qrefs], [('buildsets', None, 'complete')])
        self.assertFalse(self.fired)
        self.sendCompletionMessage(200)
        self.reactor.advance(0)
        self.assertEqual([q.filter for q in sched.master.mq.qrefs], [])
        self.assertTrue(self.fired)
        return d

    def test_trigger_overlapping(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(codebases={'cb': {'repository': 'r'}})
        self.assertEqual(sched.master.mq.qrefs, [])
        waited_for = False

        def makeSS(rev):
            if False:
                while True:
                    i = 10
            return {'revision': rev, 'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb'}
        (idsDeferred, d) = sched.trigger(waited_for, [makeSS('myrev1')])
        self.assertTriggeredBuildset(idsDeferred, waited_for, sourcestamps=[{'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb', 'revision': 'myrev1'}])
        d.addCallback(lambda res_brids: self.assertEqual(res_brids[0], 11) and self.assertEqual(res_brids[1], {77: 1000}))
        waited_for = True
        (idsDeferred, d) = sched.trigger(waited_for, [makeSS('myrev2')])
        self.reactor.advance(0)
        self.assertTriggeredBuildset(idsDeferred, waited_for, sourcestamps=[{'branch': 'br', 'project': 'p', 'repository': 'r', 'codebase': 'cb', 'revision': 'myrev2'}])
        d.addCallback(lambda res_brids1: self.assertEqual(res_brids1[0], 22) and self.assertEqual(res_brids1[1], {77: 1001}))
        self.assertEqual([q.filter for q in sched.master.mq.qrefs], [('buildsets', None, 'complete')])
        self.sendCompletionMessage(29, results=3)
        self.sendCompletionMessage(201, results=22)
        self.sendCompletionMessage(9, results=3)
        self.sendCompletionMessage(200, results=11)
        self.reactor.advance(0)
        self.assertEqual(sched.master.mq.qrefs, [])

    @defer.inlineCallbacks
    def test_trigger_with_sourcestamp(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(overrideBuildsetMethods=True)
        waited_for = False
        ss = {'repository': 'r3', 'codebase': 'cb3', 'revision': 'fixrev3', 'branch': 'default', 'project': 'p'}
        idsDeferred = sched.trigger(waited_for, sourcestamps=[ss])[0]
        yield idsDeferred
        self.assertEqual(self.addBuildsetCalls, [('addBuildsetForSourceStampsWithDefaults', {'builderNames': None, 'priority': None, 'properties': {'scheduler': ('n', 'Scheduler')}, 'reason': "The Triggerable scheduler named 'n' triggered this build", 'sourcestamps': [{'branch': 'default', 'codebase': 'cb3', 'project': 'p', 'repository': 'r3', 'revision': 'fixrev3'}], 'waited_for': False})])

    @defer.inlineCallbacks
    def test_trigger_without_sourcestamps(self):
        if False:
            i = 10
            return i + 15
        waited_for = True
        sched = self.makeScheduler(overrideBuildsetMethods=True)
        idsDeferred = sched.trigger(waited_for, sourcestamps=[])[0]
        yield idsDeferred
        self.assertEqual(self.addBuildsetCalls, [('addBuildsetForSourceStampsWithDefaults', {'builderNames': None, 'priority': None, 'properties': {'scheduler': ('n', 'Scheduler')}, 'reason': "The Triggerable scheduler named 'n' triggered this build", 'sourcestamps': [], 'waited_for': True})])

    @defer.inlineCallbacks
    def test_trigger_with_reason(self):
        if False:
            for i in range(10):
                print('nop')
        waited_for = True
        sched = self.makeScheduler(overrideBuildsetMethods=True)
        set_props = properties.Properties()
        set_props.setProperty('reason', 'test1', 'test')
        (idsDeferred, _) = sched.trigger(waited_for, sourcestamps=[], set_props=set_props)
        yield idsDeferred
        self.assertEqual(self.addBuildsetCalls, [('addBuildsetForSourceStampsWithDefaults', {'builderNames': None, 'priority': None, 'properties': {'scheduler': ('n', 'Scheduler'), 'reason': ('test1', 'test')}, 'reason': 'test1', 'sourcestamps': [], 'waited_for': True})])

    @defer.inlineCallbacks
    def test_startService_stopService(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler()
        yield sched.startService()
        yield sched.stopService()