from twisted.internet import defer
from buildbot.db import changesources
from buildbot.test.fakedb.base import FakeDBComponent
from buildbot.test.fakedb.row import Row

class ChangeSource(Row):
    table = 'changesources'
    id_column = 'id'
    hashedColumns = [('name_hash', ('name',))]

    def __init__(self, id=None, name='csname', name_hash=None):
        if False:
            while True:
                i = 10
        super().__init__(id=id, name=name, name_hash=name_hash)

class ChangeSourceMaster(Row):
    table = 'changesource_masters'
    foreignKeys = ('changesourceid', 'masterid')
    required_columns = ('changesourceid', 'masterid')

    def __init__(self, changesourceid=None, masterid=None):
        if False:
            i = 10
            return i + 15
        super().__init__(changesourceid=changesourceid, masterid=masterid)

class FakeChangeSourcesComponent(FakeDBComponent):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.changesources = {}
        self.changesource_masters = {}
        self.states = {}

    def insert_test_data(self, rows):
        if False:
            return 10
        for row in rows:
            if isinstance(row, ChangeSource):
                self.changesources[row.id] = row.name
            if isinstance(row, ChangeSourceMaster):
                self.changesource_masters[row.changesourceid] = row.masterid

    def findChangeSourceId(self, name):
        if False:
            for i in range(10):
                print('nop')
        for (cs_id, cs_name) in self.changesources.items():
            if cs_name == name:
                return defer.succeed(cs_id)
        new_id = max(self.changesources) + 1 if self.changesources else 1
        self.changesources[new_id] = name
        return defer.succeed(new_id)

    def getChangeSource(self, changesourceid):
        if False:
            return 10
        if changesourceid in self.changesources:
            rv = {'id': changesourceid, 'name': self.changesources[changesourceid], 'masterid': None}
            rv['masterid'] = self.changesource_masters.get(changesourceid)
            return defer.succeed(rv)
        return None

    def getChangeSources(self, active=None, masterid=None):
        if False:
            for i in range(10):
                print('nop')
        d = defer.DeferredList([self.getChangeSource(id) for id in self.changesources])

        @d.addCallback
        def filter(results):
            if False:
                i = 10
                return i + 15
            results = [r[1] for r in results]
            if masterid is not None:
                results = [r for r in results if r['masterid'] == masterid]
            if active:
                results = [r for r in results if r['masterid'] is not None]
            elif active is not None:
                results = [r for r in results if r['masterid'] is None]
            return results
        return d

    def setChangeSourceMaster(self, changesourceid, masterid):
        if False:
            i = 10
            return i + 15
        current_masterid = self.changesource_masters.get(changesourceid)
        if current_masterid and masterid is not None and (current_masterid != masterid):
            return defer.fail(changesources.ChangeSourceAlreadyClaimedError())
        self.changesource_masters[changesourceid] = masterid
        return defer.succeed(None)

    def fakeChangeSource(self, name, changesourceid):
        if False:
            i = 10
            return i + 15
        self.changesources[changesourceid] = name

    def fakeChangeSourceMaster(self, changesourceid, masterid):
        if False:
            for i in range(10):
                print('nop')
        if masterid is not None:
            self.changesource_masters[changesourceid] = masterid
        else:
            del self.changesource_masters[changesourceid]

    def assertChangeSourceMaster(self, changesourceid, masterid):
        if False:
            print('Hello World!')
        self.t.assertEqual(self.changesource_masters.get(changesourceid), masterid)