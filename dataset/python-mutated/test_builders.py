from twisted.internet import defer
from twisted.trial import unittest
from buildbot.db import builders
from buildbot.db import tags
from buildbot.test import fakedb
from buildbot.test.util import connector_component
from buildbot.test.util import interfaces
from buildbot.test.util import validation

def builderKey(builder):
    if False:
        print('Hello World!')
    return builder['id']

class Tests(interfaces.InterfaceTests):
    builder_row = [fakedb.Builder(id=7, name='some:builder')]

    def test_signature_findBuilderId(self):
        if False:
            return 10

        @self.assertArgSpecMatches(self.db.builders.findBuilderId)
        def findBuilderId(self, name, autoCreate=True):
            if False:
                return 10
            pass

    def test_signature_addBuilderMaster(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.db.builders.addBuilderMaster)
        def addBuilderMaster(self, builderid=None, masterid=None):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_removeBuilderMaster(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.db.builders.removeBuilderMaster)
        def removeBuilderMaster(self, builderid=None, masterid=None):
            if False:
                print('Hello World!')
            pass

    def test_signature_getBuilder(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.db.builders.getBuilder)
        def getBuilder(self, builderid):
            if False:
                return 10
            pass

    def test_signature_getBuilders(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.db.builders.getBuilders)
        def getBuilders(self, masterid=None, projectid=None):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_updateBuilderInfo(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.db.builders.updateBuilderInfo)
        def updateBuilderInfo(self, builderid, description, description_format, description_html, projectid, tags):
            if False:
                while True:
                    i = 10
            pass

    @defer.inlineCallbacks
    def test_updateBuilderInfo(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Project(id=123, name='fake_project123'), fakedb.Project(id=124, name='fake_project124'), fakedb.Builder(id=7, name='some:builder7'), fakedb.Builder(id=8, name='some:builder8')])
        yield self.db.builders.updateBuilderInfo(7, 'a string which describe the builder', None, None, 123, ['cat1', 'cat2'])
        yield self.db.builders.updateBuilderInfo(8, 'a string which describe the builder', None, None, 124, [])
        builderdict7 = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict7)
        builderdict7['tags'].sort()
        self.assertEqual(builderdict7, {'id': 7, 'name': 'some:builder7', 'tags': ['cat1', 'cat2'], 'masterids': [], 'description': 'a string which describe the builder', 'description_format': None, 'description_html': None, 'projectid': 123})
        builderdict8 = (yield self.db.builders.getBuilder(8))
        validation.verifyDbDict(self, 'builderdict', builderdict8)
        self.assertEqual(builderdict8, {'id': 8, 'name': 'some:builder8', 'tags': [], 'masterids': [], 'description': 'a string which describe the builder', 'description_format': None, 'description_html': None, 'projectid': 124})

    @defer.inlineCallbacks
    def test_update_builder_info_tags_case(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Project(id=107, name='fake_project'), fakedb.Builder(id=7, name='some:builder7', projectid=107)])
        yield self.db.builders.updateBuilderInfo(7, 'builder_desc', None, None, 107, ['Cat', 'cat'])
        builder_dict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builder_dict)
        builder_dict['tags'].sort()
        self.assertEqual(builder_dict, {'id': 7, 'name': 'some:builder7', 'tags': ['Cat', 'cat'], 'masterids': [], 'description': 'builder_desc', 'description_format': None, 'description_html': None, 'projectid': 107})

    @defer.inlineCallbacks
    def test_findBuilderId_new(self):
        if False:
            i = 10
            return i + 15
        id = (yield self.db.builders.findBuilderId('some:builder'))
        builderdict = (yield self.db.builders.getBuilder(id))
        self.assertEqual(builderdict, {'id': id, 'name': 'some:builder', 'tags': [], 'masterids': [], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_findBuilderId_new_no_autoCreate(self):
        if False:
            print('Hello World!')
        id = (yield self.db.builders.findBuilderId('some:builder', autoCreate=False))
        self.assertIsNone(id)

    @defer.inlineCallbacks
    def test_findBuilderId_exists(self):
        if False:
            return 10
        yield self.insert_test_data([fakedb.Builder(id=7, name='some:builder')])
        id = (yield self.db.builders.findBuilderId('some:builder'))
        self.assertEqual(id, 7)

    @defer.inlineCallbacks
    def test_addBuilderMaster(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Builder(id=7), fakedb.Master(id=9, name='abc'), fakedb.Master(id=10, name='def'), fakedb.BuilderMaster(builderid=7, masterid=10)])
        yield self.db.builders.addBuilderMaster(builderid=7, masterid=9)
        builderdict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(builderdict, {'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [9, 10], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_addBuilderMaster_already_present(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Builder(id=7), fakedb.Master(id=9, name='abc'), fakedb.Master(id=10, name='def'), fakedb.BuilderMaster(builderid=7, masterid=9)])
        yield self.db.builders.addBuilderMaster(builderid=7, masterid=9)
        builderdict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(builderdict, {'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [9], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_removeBuilderMaster(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.Builder(id=7), fakedb.Master(id=9, name='some:master'), fakedb.Master(id=10, name='other:master'), fakedb.BuilderMaster(builderid=7, masterid=9), fakedb.BuilderMaster(builderid=7, masterid=10)])
        yield self.db.builders.removeBuilderMaster(builderid=7, masterid=9)
        builderdict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(builderdict, {'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [10], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_getBuilder_no_masters(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Builder(id=7, name='some:builder')])
        builderdict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(builderdict, {'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_getBuilder_with_masters(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Builder(id=7, name='some:builder'), fakedb.Master(id=3, name='m1'), fakedb.Master(id=4, name='m2'), fakedb.BuilderMaster(builderid=7, masterid=3), fakedb.BuilderMaster(builderid=7, masterid=4)])
        builderdict = (yield self.db.builders.getBuilder(7))
        validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(builderdict, {'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [3, 4], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None})

    @defer.inlineCallbacks
    def test_getBuilder_missing(self):
        if False:
            return 10
        builderdict = (yield self.db.builders.getBuilder(7))
        self.assertEqual(builderdict, None)

    @defer.inlineCallbacks
    def test_getBuilders(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Builder(id=7, name='some:builder'), fakedb.Builder(id=8, name='other:builder'), fakedb.Builder(id=9, name='third:builder'), fakedb.Master(id=3, name='m1'), fakedb.Master(id=4, name='m2'), fakedb.BuilderMaster(builderid=7, masterid=3), fakedb.BuilderMaster(builderid=8, masterid=3), fakedb.BuilderMaster(builderid=8, masterid=4)])
        builderlist = (yield self.db.builders.getBuilders())
        for builderdict in builderlist:
            validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(sorted(builderlist, key=builderKey), sorted([{'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [3], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None}, {'id': 8, 'name': 'other:builder', 'tags': [], 'masterids': [3, 4], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None}, {'id': 9, 'name': 'third:builder', 'tags': [], 'masterids': [], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None}], key=builderKey))

    @defer.inlineCallbacks
    def test_getBuilders_masterid(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.insert_test_data([fakedb.Builder(id=7, name='some:builder'), fakedb.Builder(id=8, name='other:builder'), fakedb.Builder(id=9, name='third:builder'), fakedb.Master(id=3, name='m1'), fakedb.Master(id=4, name='m2'), fakedb.BuilderMaster(builderid=7, masterid=3), fakedb.BuilderMaster(builderid=8, masterid=3), fakedb.BuilderMaster(builderid=8, masterid=4)])
        builderlist = (yield self.db.builders.getBuilders(masterid=3))
        for builderdict in builderlist:
            validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(sorted(builderlist, key=builderKey), sorted([{'id': 7, 'name': 'some:builder', 'tags': [], 'masterids': [3], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None}, {'id': 8, 'name': 'other:builder', 'tags': [], 'masterids': [3, 4], 'description': None, 'description_format': None, 'description_html': None, 'projectid': None}], key=builderKey))

    @defer.inlineCallbacks
    def test_getBuilders_projectid(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Project(id=201, name='p201'), fakedb.Project(id=202, name='p202'), fakedb.Builder(id=101, name='b101'), fakedb.Builder(id=102, name='b102', projectid=201), fakedb.Builder(id=103, name='b103', projectid=201), fakedb.Builder(id=104, name='b104', projectid=202), fakedb.Master(id=3, name='m1'), fakedb.Master(id=4, name='m2'), fakedb.BuilderMaster(builderid=101, masterid=3), fakedb.BuilderMaster(builderid=102, masterid=3), fakedb.BuilderMaster(builderid=103, masterid=4), fakedb.BuilderMaster(builderid=104, masterid=4)])
        builderlist = (yield self.db.builders.getBuilders(projectid=201))
        for builderdict in builderlist:
            validation.verifyDbDict(self, 'builderdict', builderdict)
        self.assertEqual(sorted(builderlist, key=builderKey), sorted([{'id': 102, 'name': 'b102', 'masterids': [3], 'tags': [], 'description': None, 'description_format': None, 'description_html': None, 'projectid': 201}, {'id': 103, 'name': 'b103', 'masterids': [4], 'tags': [], 'description': None, 'description_format': None, 'description_html': None, 'projectid': 201}], key=builderKey))

    @defer.inlineCallbacks
    def test_getBuilders_empty(self):
        if False:
            for i in range(10):
                print('nop')
        builderlist = (yield self.db.builders.getBuilders())
        self.assertEqual(sorted(builderlist), [])

class RealTests(Tests):
    pass

class TestFakeDB(unittest.TestCase, connector_component.FakeConnectorComponentMixin, Tests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpConnectorComponent()

class TestRealDB(unittest.TestCase, connector_component.ConnectorComponentMixin, RealTests):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        yield self.setUpConnectorComponent(table_names=['projects', 'builders', 'masters', 'builder_masters', 'builders_tags', 'tags'])
        self.db.builders = builders.BuildersConnectorComponent(self.db)
        self.db.tags = tags.TagsConnectorComponent(self.db)
        self.master = self.db.master
        self.master.db = self.db

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tearDownConnectorComponent()