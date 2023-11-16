import json
from twisted.internet import defer
from buildbot.db import buildsets
from buildbot.test.fakedb.base import FakeDBComponent
from buildbot.test.fakedb.buildrequests import BuildRequest
from buildbot.test.fakedb.row import Row
from buildbot.util import datetime2epoch
from buildbot.util import epoch2datetime

class Buildset(Row):
    table = 'buildsets'
    id_column = 'id'

    def __init__(self, id=None, external_idstring='extid', reason='because', submitted_at=12345678, complete=0, complete_at=None, results=-1, parent_buildid=None, parent_relationship=None):
        if False:
            print('Hello World!')
        super().__init__(id=id, external_idstring=external_idstring, reason=reason, submitted_at=submitted_at, complete=complete, complete_at=complete_at, results=results, parent_buildid=parent_buildid, parent_relationship=parent_relationship)

class BuildsetProperty(Row):
    table = 'buildset_properties'
    foreignKeys = ('buildsetid',)
    required_columns = ('buildsetid',)

    def __init__(self, buildsetid=None, property_name='prop', property_value='[22, "fakedb"]'):
        if False:
            while True:
                i = 10
        super().__init__(buildsetid=buildsetid, property_name=property_name, property_value=property_value)

class BuildsetSourceStamp(Row):
    table = 'buildset_sourcestamps'
    foreignKeys = ('buildsetid', 'sourcestampid')
    required_columns = ('buildsetid', 'sourcestampid')
    id_column = 'id'

    def __init__(self, id=None, buildsetid=None, sourcestampid=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(id=id, buildsetid=buildsetid, sourcestampid=sourcestampid)

class FakeBuildsetsComponent(FakeDBComponent):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.buildsets = {}
        self.completed_bsids = set()
        self.buildset_sourcestamps = {}

    def insert_test_data(self, rows):
        if False:
            i = 10
            return i + 15
        for row in rows:
            if isinstance(row, Buildset):
                bs = self.buildsets[row.id] = row.values.copy()
                bs['properties'] = {}
        for row in rows:
            if isinstance(row, BuildsetProperty):
                assert row.buildsetid in self.buildsets
                n = row.property_name
                (v, src) = tuple(json.loads(row.property_value))
                self.buildsets[row.buildsetid]['properties'][n] = (v, src)
        for row in rows:
            if isinstance(row, BuildsetSourceStamp):
                assert row.buildsetid in self.buildsets
                self.buildset_sourcestamps.setdefault(row.buildsetid, []).append(row.sourcestampid)

    def _newBsid(self):
        if False:
            print('Hello World!')
        bsid = 200
        while bsid in self.buildsets:
            bsid += 1
        return bsid

    @defer.inlineCallbacks
    def addBuildset(self, sourcestamps, reason, properties, builderids, waited_for, external_idstring=None, submitted_at=None, parent_buildid=None, parent_relationship=None, priority=0):
        if False:
            print('Hello World!')
        assert isinstance(waited_for, bool), f'waited_for should be boolean: {repr(waited_for)}'
        if submitted_at is not None:
            submitted_at = datetime2epoch(submitted_at)
        else:
            submitted_at = int(self.reactor.seconds())
        bsid = self._newBsid()
        br_rows = []
        for builderid in builderids:
            br_rows.append(BuildRequest(buildsetid=bsid, builderid=builderid, waited_for=waited_for, submitted_at=submitted_at))
        self.db.buildrequests.insert_test_data(br_rows)
        bsrow = Buildset(id=bsid, reason=reason, external_idstring=external_idstring, submitted_at=submitted_at, parent_buildid=parent_buildid, parent_relationship=parent_relationship)
        self.buildsets[bsid] = bsrow.values.copy()
        self.buildsets[bsid]['properties'] = properties
        ssids = []
        for ss in sourcestamps:
            if not isinstance(ss, type(1)):
                ss = (yield self.db.sourcestamps.findSourceStampId(**ss))
            ssids.append(ss)
        self.buildset_sourcestamps[bsid] = ssids
        return (bsid, {br.builderid: br.id for br in br_rows})

    def completeBuildset(self, bsid, results, complete_at=None):
        if False:
            for i in range(10):
                print('nop')
        if bsid not in self.buildsets or self.buildsets[bsid]['complete']:
            raise buildsets.AlreadyCompleteError()
        if complete_at is not None:
            complete_at = datetime2epoch(complete_at)
        else:
            complete_at = int(self.reactor.seconds())
        self.buildsets[bsid]['results'] = results
        self.buildsets[bsid]['complete'] = 1
        self.buildsets[bsid]['complete_at'] = complete_at
        return defer.succeed(None)

    def getBuildset(self, bsid):
        if False:
            print('Hello World!')
        if bsid not in self.buildsets:
            return defer.succeed(None)
        row = self.buildsets[bsid]
        return defer.succeed(self._row2dict(row))

    def getBuildsets(self, complete=None, resultSpec=None):
        if False:
            i = 10
            return i + 15
        rv = []
        for bs in self.buildsets.values():
            if complete is not None:
                if complete and bs['complete']:
                    rv.append(bs)
                elif not complete and (not bs['complete']):
                    rv.append(bs)
            else:
                rv.append(bs)
        if resultSpec is not None:
            rv = self.applyResultSpec(rv, resultSpec)
        rv = [self._row2dict(bs) for bs in rv]
        return defer.succeed(rv)

    @defer.inlineCallbacks
    def getRecentBuildsets(self, count=None, branch=None, repository=None, complete=None):
        if False:
            return 10
        if not count:
            return []
        rv = []
        for bs in (yield self.getBuildsets(complete=complete)):
            if branch or repository:
                ok = True
                if not bs['sourcestamps']:
                    ok = False
                for ssid in bs['sourcestamps']:
                    ss = (yield self.db.sourcestamps.getSourceStamp(ssid))
                    if branch and ss['branch'] != branch:
                        ok = False
                    if repository and ss['repository'] != repository:
                        ok = False
            else:
                ok = True
            if ok:
                rv.append(bs)
        rv.sort(key=lambda bs: -bs['bsid'])
        return list(reversed(rv[:count]))

    def _row2dict(self, row):
        if False:
            i = 10
            return i + 15
        row = row.copy()
        row['complete_at'] = epoch2datetime(row['complete_at'])
        row['submitted_at'] = epoch2datetime(row['submitted_at'])
        row['complete'] = bool(row['complete'])
        row['bsid'] = row['id']
        row['sourcestamps'] = self.buildset_sourcestamps.get(row['id'], [])
        del row['id']
        del row['properties']
        return row

    def getBuildsetProperties(self, key, no_cache=False):
        if False:
            i = 10
            return i + 15
        if key in self.buildsets:
            return defer.succeed(self.buildsets[key]['properties'])
        return defer.succeed({})

    def fakeBuildsetCompletion(self, bsid, result):
        if False:
            print('Hello World!')
        assert bsid in self.buildsets
        self.buildsets[bsid]['results'] = result
        self.completed_bsids.add(bsid)

    def assertBuildsetCompletion(self, bsid, complete):
        if False:
            for i in range(10):
                print('nop')
        'Assert that the completion state of buildset BSID is COMPLETE'
        actual = self.buildsets[bsid]['complete']
        self.t.assertTrue(actual and complete or (not actual and (not complete)))

    def assertBuildset(self, bsid=None, expected_buildset=None):
        if False:
            for i in range(10):
                print('nop')
        'Assert that the given buildset looks as expected; the ssid parameter\n        of the buildset is omitted.  Properties are converted with asList and\n        sorted.  Attributes complete, complete_at, submitted_at, results, and parent_*\n        are ignored if not specified.'
        self.t.assertIn(bsid, self.buildsets)
        buildset = self.buildsets[bsid].copy()
        del buildset['id']
        columns = ['complete', 'complete_at', 'submitted_at', 'results', 'parent_buildid', 'parent_relationship']
        for col in columns:
            if col not in expected_buildset:
                del buildset[col]
        if buildset['properties']:
            buildset['properties'] = sorted(buildset['properties'].items())
        self.t.assertEqual(buildset, expected_buildset)
        return bsid