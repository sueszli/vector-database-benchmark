from twisted.internet import defer
from twisted.trial import unittest
from buildbot.db import changes
from buildbot.test import fakedb
from buildbot.test.util import connector_component

class TestBadRows(connector_component.ConnectorComponentMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.setUpConnectorComponent(table_names=['patches', 'sourcestamps', 'changes', 'change_properties', 'change_files'])
        self.db.changes = changes.ChangesConnectorComponent(self.db)

    def tearDown(self):
        if False:
            print('Hello World!')
        return self.tearDownConnectorComponent()

    @defer.inlineCallbacks
    def test_bogus_row_no_source(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.SourceStamp(id=10), fakedb.ChangeProperty(changeid=13, property_name='devel', property_value='"no source"'), fakedb.Change(changeid=13, sourcestampid=10)])
        c = (yield self.db.changes.getChange(13))
        self.assertEqual(c['properties'], {'devel': ('no source', 'Change')})

    @defer.inlineCallbacks
    def test_bogus_row_jsoned_list(self):
        if False:
            print('Hello World!')
        yield self.insert_test_data([fakedb.SourceStamp(id=10), fakedb.ChangeProperty(changeid=13, property_name='devel', property_value='[1, 2]'), fakedb.Change(changeid=13, sourcestampid=10)])
        c = (yield self.db.changes.getChange(13))
        self.assertEqual(c['properties'], {'devel': ([1, 2], 'Change')})