from twisted.internet import defer
from twisted.trial import unittest
from buildbot.db import masters
from buildbot.test import fakedb
from buildbot.test.util import connector_component
from buildbot.test.util import interfaces
from buildbot.test.util import validation
from buildbot.util import epoch2datetime
SOMETIME = 1348971992
SOMETIME_DT = epoch2datetime(SOMETIME)
OTHERTIME = 1008971992
OTHERTIME_DT = epoch2datetime(OTHERTIME)

class Tests(interfaces.InterfaceTests):
    master_row = [fakedb.Master(id=7, name='some:master', active=1, last_active=SOMETIME)]

    def test_signature_findMasterId(self):
        if False:
            for i in range(10):
                print('nop')

        @self.assertArgSpecMatches(self.db.masters.findMasterId)
        def findMasterId(self, name):
            if False:
                print('Hello World!')
            pass

    def test_signature_setMasterState(self):
        if False:
            for i in range(10):
                print('nop')

        @self.assertArgSpecMatches(self.db.masters.setMasterState)
        def setMasterState(self, masterid, active):
            if False:
                print('Hello World!')
            pass

    def test_signature_getMaster(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.db.masters.getMaster)
        def getMaster(self, masterid):
            if False:
                print('Hello World!')
            pass

    def test_signature_getMasters(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.db.masters.getMasters)
        def getMasters(self):
            if False:
                print('Hello World!')
            pass

    @defer.inlineCallbacks
    def test_findMasterId_new(self):
        if False:
            i = 10
            return i + 15
        id = (yield self.db.masters.findMasterId('some:master'))
        masterdict = (yield self.db.masters.getMaster(id))
        self.assertEqual(masterdict, {'id': id, 'name': 'some:master', 'active': False, 'last_active': SOMETIME_DT})

    @defer.inlineCallbacks
    def test_findMasterId_new_name_differs_only_by_case(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master')])
        id = (yield self.db.masters.findMasterId('some:Master'))
        masterdict = (yield self.db.masters.getMaster(id))
        self.assertEqual(masterdict, {'id': id, 'name': 'some:Master', 'active': False, 'last_active': SOMETIME_DT})

    @defer.inlineCallbacks
    def test_findMasterId_exists(self):
        if False:
            return 10
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master')])
        id = (yield self.db.masters.findMasterId('some:master'))
        self.assertEqual(id, 7)

    @defer.inlineCallbacks
    def test_setMasterState_when_missing(self):
        if False:
            for i in range(10):
                print('nop')
        activated = (yield self.db.masters.setMasterState(masterid=7, active=True))
        self.assertFalse(activated)

    @defer.inlineCallbacks
    def test_setMasterState_true_when_active(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=1, last_active=OTHERTIME)])
        activated = (yield self.db.masters.setMasterState(masterid=7, active=True))
        self.assertFalse(activated)
        masterdict = (yield self.db.masters.getMaster(7))
        self.assertEqual(masterdict, {'id': 7, 'name': 'some:master', 'active': True, 'last_active': SOMETIME_DT})

    @defer.inlineCallbacks
    def test_setMasterState_true_when_inactive(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=0, last_active=OTHERTIME)])
        activated = (yield self.db.masters.setMasterState(masterid=7, active=True))
        self.assertTrue(activated)
        masterdict = (yield self.db.masters.getMaster(7))
        self.assertEqual(masterdict, {'id': 7, 'name': 'some:master', 'active': True, 'last_active': SOMETIME_DT})

    @defer.inlineCallbacks
    def test_setMasterState_false_when_active(self):
        if False:
            i = 10
            return i + 15
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=1, last_active=OTHERTIME)])
        deactivated = (yield self.db.masters.setMasterState(masterid=7, active=False))
        self.assertTrue(deactivated)
        masterdict = (yield self.db.masters.getMaster(7))
        self.assertEqual(masterdict, {'id': 7, 'name': 'some:master', 'active': False, 'last_active': OTHERTIME_DT})

    @defer.inlineCallbacks
    def test_setMasterState_false_when_inactive(self):
        if False:
            i = 10
            return i + 15
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=0, last_active=OTHERTIME)])
        deactivated = (yield self.db.masters.setMasterState(masterid=7, active=False))
        self.assertFalse(deactivated)
        masterdict = (yield self.db.masters.getMaster(7))
        self.assertEqual(masterdict, {'id': 7, 'name': 'some:master', 'active': False, 'last_active': OTHERTIME_DT})

    @defer.inlineCallbacks
    def test_getMaster(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=0, last_active=SOMETIME)])
        masterdict = (yield self.db.masters.getMaster(7))
        validation.verifyDbDict(self, 'masterdict', masterdict)
        self.assertEqual(masterdict, {'id': 7, 'name': 'some:master', 'active': False, 'last_active': SOMETIME_DT})

    @defer.inlineCallbacks
    def test_getMaster_missing(self):
        if False:
            return 10
        masterdict = (yield self.db.masters.getMaster(7))
        self.assertEqual(masterdict, None)

    @defer.inlineCallbacks
    def test_getMasters(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=0, last_active=SOMETIME), fakedb.Master(id=8, name='other:master', active=1, last_active=OTHERTIME)])
        masterlist = (yield self.db.masters.getMasters())
        for masterdict in masterlist:
            validation.verifyDbDict(self, 'masterdict', masterdict)

        def masterKey(master):
            if False:
                for i in range(10):
                    print('nop')
            return master['id']
        expected = sorted([{'id': 7, 'name': 'some:master', 'active': 0, 'last_active': SOMETIME_DT}, {'id': 8, 'name': 'other:master', 'active': 1, 'last_active': OTHERTIME_DT}], key=masterKey)
        self.assertEqual(sorted(masterlist, key=masterKey), expected)

class RealTests(Tests):

    @defer.inlineCallbacks
    def test_setMasterState_false_deletes_links(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=1, last_active=OTHERTIME), fakedb.Scheduler(id=21), fakedb.SchedulerMaster(schedulerid=21, masterid=7)])
        deactivated = (yield self.db.masters.setMasterState(masterid=7, active=False))
        self.assertTrue(deactivated)

        def thd(conn):
            if False:
                for i in range(10):
                    print('nop')
            tbl = self.db.model.scheduler_masters
            self.assertEqual(conn.execute(tbl.select()).fetchall(), [])
        yield self.db.pool.do(thd)

class TestFakeDB(unittest.TestCase, connector_component.FakeConnectorComponentMixin, Tests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpConnectorComponent()
        self.reactor.advance(SOMETIME)

class TestRealDB(unittest.TestCase, connector_component.ConnectorComponentMixin, RealTests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        yield self.setUpConnectorComponent(table_names=['masters', 'schedulers', 'scheduler_masters'])
        self.reactor.advance(SOMETIME)
        self.db.masters = masters.MastersConnectorComponent(self.db)

    def tearDown(self):
        if False:
            return 10
        return self.tearDownConnectorComponent()