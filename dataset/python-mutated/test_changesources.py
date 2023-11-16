from twisted.internet import defer
from twisted.trial import unittest
from buildbot.db import changesources
from buildbot.test import fakedb
from buildbot.test.util import connector_component
from buildbot.test.util import db
from buildbot.test.util import interfaces
from buildbot.test.util import validation

def changeSourceKey(changeSource):
    if False:
        i = 10
        return i + 15
    return changeSource['id']

class Tests(interfaces.InterfaceTests):
    cs42 = fakedb.ChangeSource(id=42, name='cool_source')
    cs87 = fakedb.ChangeSource(id=87, name='lame_source')
    master13 = fakedb.Master(id=13, name='m1', active=1)
    cs42master13 = fakedb.ChangeSourceMaster(changesourceid=42, masterid=13)
    master14 = fakedb.Master(id=14, name='m2', active=0)
    cs87master14 = fakedb.ChangeSourceMaster(changesourceid=87, masterid=14)

    def test_signature_findChangeSourceId(self):
        if False:
            return 10
        'The signature of findChangeSourceId is correct'

        @self.assertArgSpecMatches(self.db.changesources.findChangeSourceId)
        def findChangeSourceId(self, name):
            if False:
                while True:
                    i = 10
            pass

    @defer.inlineCallbacks
    def test_findChangeSourceId_new(self):
        if False:
            i = 10
            return i + 15
        'findChangeSourceId for a new changesource creates it'
        id = (yield self.db.changesources.findChangeSourceId('csname'))
        cs = (yield self.db.changesources.getChangeSource(id))
        self.assertEqual(cs['name'], 'csname')

    @defer.inlineCallbacks
    def test_findChangeSourceId_existing(self):
        if False:
            while True:
                i = 10
        'findChangeSourceId gives the same answer for the same inputs'
        id1 = (yield self.db.changesources.findChangeSourceId('csname'))
        id2 = (yield self.db.changesources.findChangeSourceId('csname'))
        self.assertEqual(id1, id2)

    def test_signature_setChangeSourceMaster(self):
        if False:
            return 10
        'setChangeSourceMaster has the right signature'

        @self.assertArgSpecMatches(self.db.changesources.setChangeSourceMaster)
        def setChangeSourceMaster(self, changesourceid, masterid):
            if False:
                i = 10
                return i + 15
            pass

    @defer.inlineCallbacks
    def test_setChangeSourceMaster_fresh(self):
        if False:
            for i in range(10):
                print('nop')
        'setChangeSourceMaster with a good pair'
        yield self.insert_test_data([self.cs42, self.master13])
        yield self.db.changesources.setChangeSourceMaster(42, 13)
        cs = (yield self.db.changesources.getChangeSource(42))
        self.assertEqual(cs['masterid'], 13)

    @defer.inlineCallbacks
    def test_setChangeSourceMaster_inactive_but_linked(self):
        if False:
            print('Hello World!')
        'Inactive changesource but already claimed by an active master'
        d = self.insert_test_data([self.cs87, self.master13, self.master14, self.cs87master14])
        d.addCallback(lambda _: self.db.changesources.setChangeSourceMaster(87, 13))
        yield self.assertFailure(d, changesources.ChangeSourceAlreadyClaimedError)

    @defer.inlineCallbacks
    def test_setChangeSourceMaster_active(self):
        if False:
            print('Hello World!')
        'Active changesource already claimed by an active master'
        d = self.insert_test_data([self.cs42, self.master13, self.cs42master13])
        d.addCallback(lambda _: self.db.changesources.setChangeSourceMaster(42, 14))
        yield self.assertFailure(d, changesources.ChangeSourceAlreadyClaimedError)

    @defer.inlineCallbacks
    def test_setChangeSourceMaster_None(self):
        if False:
            print('Hello World!')
        "A 'None' master disconnects the changesource"
        yield self.insert_test_data([self.cs87, self.master14, self.cs87master14])
        yield self.db.changesources.setChangeSourceMaster(87, None)
        cs = (yield self.db.changesources.getChangeSource(87))
        self.assertEqual(cs['masterid'], None)

    @defer.inlineCallbacks
    def test_setChangeSourceMaster_None_unowned(self):
        if False:
            print('Hello World!')
        "A 'None' master for a disconnected changesource"
        yield self.insert_test_data([self.cs87])
        yield self.db.changesources.setChangeSourceMaster(87, None)
        cs = (yield self.db.changesources.getChangeSource(87))
        self.assertEqual(cs['masterid'], None)

    def test_signature_getChangeSource(self):
        if False:
            while True:
                i = 10
        'getChangeSource has the right signature'

        @self.assertArgSpecMatches(self.db.changesources.getChangeSource)
        def getChangeSource(self, changesourceid):
            if False:
                i = 10
                return i + 15
            pass

    @defer.inlineCallbacks
    def test_getChangeSource(self):
        if False:
            print('Hello World!')
        'getChangeSource for a changesource that exists'
        yield self.insert_test_data([self.cs87])
        cs = (yield self.db.changesources.getChangeSource(87))
        validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(cs, {'id': 87, 'name': 'lame_source', 'masterid': None})

    @defer.inlineCallbacks
    def test_getChangeSource_missing(self):
        if False:
            while True:
                i = 10
        "getChangeSource for a changesource that doesn't exist"
        cs = (yield self.db.changesources.getChangeSource(87))
        self.assertEqual(cs, None)

    @defer.inlineCallbacks
    def test_getChangeSource_active(self):
        if False:
            return 10
        'getChangeSource for a changesource that exists and is active'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13])
        cs = (yield self.db.changesources.getChangeSource(42))
        validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(cs, {'id': 42, 'name': 'cool_source', 'masterid': 13})

    @defer.inlineCallbacks
    def test_getChangeSource_inactive_but_linked(self):
        if False:
            i = 10
            return i + 15
        'getChangeSource for a changesource that is assigned but is inactive'
        yield self.insert_test_data([self.cs87, self.master14, self.cs87master14])
        cs = (yield self.db.changesources.getChangeSource(87))
        validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(cs, {'id': 87, 'name': 'lame_source', 'masterid': 14})

    def test_signature_getChangeSources(self):
        if False:
            i = 10
            return i + 15
        'getChangeSources has right signature'

        @self.assertArgSpecMatches(self.db.changesources.getChangeSources)
        def getChangeSources(self, active=None, masterid=None):
            if False:
                print('Hello World!')
            pass

    @defer.inlineCallbacks
    def test_getChangeSources(self):
        if False:
            print('Hello World!')
        'getChangeSources returns all changesources'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources())
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist, key=changeSourceKey), sorted([{'id': 42, 'name': 'cool_source', 'masterid': 13}, {'id': 87, 'name': 'lame_source', 'masterid': None}], key=changeSourceKey))

    @defer.inlineCallbacks
    def test_getChangeSources_masterid(self):
        if False:
            while True:
                i = 10
        'getChangeSources returns all changesources for a given master'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources(masterid=13))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist, key=changeSourceKey), sorted([{'id': 42, 'name': 'cool_source', 'masterid': 13}], key=changeSourceKey))

    @defer.inlineCallbacks
    def test_getChangeSources_active(self):
        if False:
            i = 10
            return i + 15
        'getChangeSources for (active changesources, all masters)'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources(active=True))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), sorted([{'id': 42, 'name': 'cool_source', 'masterid': 13}]))

    @defer.inlineCallbacks
    def test_getChangeSources_active_masterid(self):
        if False:
            for i in range(10):
                print('nop')
        'getChangeSources returns (active changesources, given masters)'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources(active=True, masterid=13))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), sorted([{'id': 42, 'name': 'cool_source', 'masterid': 13}]))
        cslist = (yield self.db.changesources.getChangeSources(active=True, masterid=14))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), [])

    @defer.inlineCallbacks
    def test_getChangeSources_inactive(self):
        if False:
            print('Hello World!')
        'getChangeSources returns (inactive changesources, all masters)'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources(active=False))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), sorted([{'id': 87, 'name': 'lame_source', 'masterid': None}]))

    @defer.inlineCallbacks
    def test_getChangeSources_inactive_masterid(self):
        if False:
            i = 10
            return i + 15
        'getChangeSources returns (active changesources, given masters)'
        yield self.insert_test_data([self.cs42, self.master13, self.cs42master13, self.cs87])
        cslist = (yield self.db.changesources.getChangeSources(active=False, masterid=13))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), [])
        cslist = (yield self.db.changesources.getChangeSources(active=False, masterid=14))
        for cs in cslist:
            validation.verifyDbDict(self, 'changesourcedict', cs)
        self.assertEqual(sorted(cslist), [])

class RealTests(Tests):
    pass

class TestFakeDB(unittest.TestCase, connector_component.FakeConnectorComponentMixin, Tests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        yield self.setUpConnectorComponent()

class TestRealDB(db.TestCase, connector_component.ConnectorComponentMixin, RealTests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpConnectorComponent(table_names=['changes', 'changesources', 'masters', 'patches', 'sourcestamps', 'changesource_masters'])
        self.db.changesources = changesources.ChangeSourcesConnectorComponent(self.db)

    def tearDown(self):
        if False:
            return 10
        return self.tearDownConnectorComponent()