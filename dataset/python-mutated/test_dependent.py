from twisted.internet import defer
from twisted.trial import unittest
from buildbot import config
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.schedulers import base
from buildbot.schedulers import dependent
from buildbot.test import fakedb
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler
SUBMITTED_AT_TIME = 111111111
COMPLETE_AT_TIME = 222222222
OBJECTID = 33
SCHEDULERID = 133
UPSTREAM_NAME = 'uppy'

class Dependent(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.setUpScheduler()

    def tearDown(self):
        if False:
            return 10
        self.tearDownScheduler()

    def makeScheduler(self, upstream=None):
        if False:
            i = 10
            return i + 15

        class Upstream(base.BaseScheduler):

            def __init__(self, name):
                if False:
                    print('Hello World!')
                self.name = name
        if not upstream:
            upstream = Upstream(UPSTREAM_NAME)
        sched = dependent.Dependent(name='n', builderNames=['b'], upstream=upstream)
        self.attachScheduler(sched, OBJECTID, SCHEDULERID, overrideBuildsetMethods=True, createBuilderDB=True)
        return sched

    def assertBuildsetSubscriptions(self, bsids=None):
        if False:
            return 10
        self.db.state.assertState(OBJECTID, upstream_bsids=bsids)

    def test_constructor_string_arg(self):
        if False:
            return 10
        with self.assertRaises(config.ConfigErrors):
            self.makeScheduler(upstream='foo')

    @defer.inlineCallbacks
    def test_activate(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler()
        sched.activate()
        self.assertEqual(sorted([q.filter for q in sched.master.mq.qrefs]), [('buildsets', None, 'complete'), ('buildsets', None, 'new'), ('schedulers', '133', 'updated')])
        yield sched.deactivate()
        self.assertEqual([q.filter for q in sched.master.mq.qrefs], [('schedulers', '133', 'updated')])

    def sendBuildsetMessage(self, scheduler_name=None, results=-1, complete=False):
        if False:
            i = 10
            return i + 15
        'Call callConsumer with a buildset message.  Most of the values here\n        are hard-coded to correspond to those in do_test.'
        msg = {'bsid': 44, 'sourcestamps': [], 'submitted_at': SUBMITTED_AT_TIME, 'complete': complete, 'complete_at': COMPLETE_AT_TIME if complete else None, 'external_idstring': None, 'reason': 'Because', 'results': results if complete else -1, 'parent_buildid': None, 'parent_relationship': None}
        if not complete:
            msg['scheduler'] = scheduler_name
        self.master.mq.callConsumer(('buildsets', '44', 'complete' if complete else 'new'), msg)

    def do_test(self, scheduler_name, expect_subscription, results, expect_buildset):
        if False:
            while True:
                i = 10
        "Test the dependent scheduler by faking a buildset and subsequent\n        completion from an upstream scheduler.\n\n        @param scheduler_name: upstream scheduler's name\n        @param expect_subscription: whether to expect the dependent to\n            subscribe to the buildset\n        @param results: results of the upstream scheduler's buildset\n        @param expect_buidlset: whether to expect the dependent to generate\n            a new buildset in response\n        "
        sched = self.makeScheduler()
        sched.activate()
        self.db.insert_test_data([fakedb.SourceStamp(id=93, revision='555', branch='master', project='proj', repository='repo', codebase='cb'), fakedb.Buildset(id=44, submitted_at=SUBMITTED_AT_TIME, complete=False, complete_at=None, external_idstring=None, reason='Because', results=-1), fakedb.BuildsetSourceStamp(buildsetid=44, sourcestampid=93)])
        self.sendBuildsetMessage(scheduler_name=scheduler_name, complete=False)
        if expect_subscription:
            self.assertBuildsetSubscriptions([44])
        else:
            self.assertBuildsetSubscriptions([])
        self.db.buildsets.fakeBuildsetCompletion(bsid=44, result=results)
        self.sendBuildsetMessage(results=results, complete=True)
        if expect_buildset:
            self.assertEqual(self.addBuildsetCalls, [('addBuildsetForSourceStamps', {'builderNames': None, 'external_idstring': None, 'properties': None, 'reason': 'downstream', 'sourcestamps': [93]})])
        else:
            self.assertEqual(self.addBuildsetCalls, [])

    def test_related_buildset_SUCCESS(self):
        if False:
            for i in range(10):
                print('nop')
        return self.do_test(UPSTREAM_NAME, True, SUCCESS, True)

    def test_related_buildset_WARNINGS(self):
        if False:
            i = 10
            return i + 15
        return self.do_test(UPSTREAM_NAME, True, WARNINGS, True)

    def test_related_buildset_FAILURE(self):
        if False:
            for i in range(10):
                print('nop')
        return self.do_test(UPSTREAM_NAME, True, FAILURE, False)

    def test_unrelated_buildset(self):
        if False:
            print('Hello World!')
        return self.do_test('unrelated', False, SUCCESS, False)

    @defer.inlineCallbacks
    def test_getUpstreamBuildsets_missing(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler()
        self.db.insert_test_data([fakedb.SourceStamp(id=1234), fakedb.Buildset(id=11), fakedb.Buildset(id=13), fakedb.BuildsetSourceStamp(buildsetid=13, sourcestampid=1234), fakedb.Object(id=OBJECTID), fakedb.ObjectState(objectid=OBJECTID, name='upstream_bsids', value_json='[11,12,13]')])
        self.assertEqual((yield sched._getUpstreamBuildsets()), [(11, [], False, -1), (13, [1234], False, -1)])
        self.db.state.assertState(OBJECTID, upstream_bsids=[11, 13])

    @defer.inlineCallbacks
    def test_enabled_callback(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler()
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
        sched = self.makeScheduler()
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, False)
        r = (yield sched.activate())
        self.assertEqual(r, None)

    @defer.inlineCallbacks
    def test_disabled_deactivate(self):
        if False:
            return 10
        sched = self.makeScheduler()
        yield sched._enabledCallback(None, {'enabled': not sched.enabled})
        self.assertEqual(sched.enabled, False)
        r = (yield sched.deactivate())
        self.assertEqual(r, None)