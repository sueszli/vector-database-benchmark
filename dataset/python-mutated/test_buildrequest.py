import datetime
import json
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process import buildrequest
from buildbot.process.builder import Builder
from buildbot.test import fakedb
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin

class TestBuildRequestCollapser(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True, wantDb=True)
        self.master.botmaster = mock.Mock(name='botmaster')
        self.master.botmaster.builders = {}
        self.builders = {}
        self.bldr = (yield self.createBuilder('A', builderid=77))

    @defer.inlineCallbacks
    def createBuilder(self, name, builderid=None):
        if False:
            while True:
                i = 10
        if builderid is None:
            b = fakedb.Builder(name=name)
            yield self.master.db.insert_test_data([b])
            builderid = b.id
        bldr = mock.Mock(name=name)
        bldr.name = name
        bldr.master = self.master
        self.master.botmaster.builders[name] = bldr
        self.builders[name] = bldr
        bldr.getCollapseRequestsFn = lambda : False
        return bldr

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    @defer.inlineCallbacks
    def do_request_collapse(self, rows, brids, exp):
        if False:
            return 10
        yield self.master.db.insert_test_data(rows)
        brCollapser = buildrequest.BuildRequestCollapser(self.master, brids)
        self.assertEqual(exp, (yield brCollapser.collapse()))

    def test_collapseRequests_no_other_request(self):
        if False:
            return 10

        def collapseRequests_fn(master, builder, brdict1, brdict2):
            if False:
                return 10
            self.fail('Should never be called')
            return True
        self.bldr.getCollapseRequestsFn = lambda : collapseRequests_fn
        rows = [fakedb.Builder(id=77, name='A'), fakedb.SourceStamp(id=234, codebase='A'), fakedb.Change(changeid=14, codebase='A', sourcestampid=234), fakedb.Buildset(id=30, reason='foo', submitted_at=1300305712, results=-1), fakedb.BuildsetSourceStamp(sourcestampid=234, buildsetid=30), fakedb.BuildRequest(id=19, buildsetid=30, builderid=77, priority=13, submitted_at=1300305712, results=-1)]
        return self.do_request_collapse(rows, [19], [])
    BASE_ROWS = [fakedb.Builder(id=77, name='A'), fakedb.SourceStamp(id=234, codebase='C'), fakedb.Buildset(id=30, reason='foo', submitted_at=1300305712, results=-1), fakedb.BuildsetSourceStamp(sourcestampid=234, buildsetid=30), fakedb.SourceStamp(id=235, codebase='C'), fakedb.Buildset(id=31, reason='foo', submitted_at=1300305712, results=-1), fakedb.BuildsetSourceStamp(sourcestampid=235, buildsetid=31), fakedb.SourceStamp(id=236, codebase='C'), fakedb.Buildset(id=32, reason='foo', submitted_at=1300305712, results=-1), fakedb.BuildsetSourceStamp(sourcestampid=236, buildsetid=32), fakedb.BuildRequest(id=19, buildsetid=30, builderid=77, priority=13, submitted_at=1300305712, results=-1), fakedb.BuildRequest(id=20, buildsetid=31, builderid=77, priority=13, submitted_at=1300305712, results=-1), fakedb.BuildRequest(id=21, buildsetid=32, builderid=77, priority=13, submitted_at=1300305712, results=-1)]

    def test_collapseRequests_no_collapse(self):
        if False:
            while True:
                i = 10

        def collapseRequests_fn(master, builder, brdict1, brdict2):
            if False:
                for i in range(10):
                    print('nop')
            return False
        self.bldr.getCollapseRequestsFn = lambda : collapseRequests_fn
        return self.do_request_collapse(self.BASE_ROWS, [21], [])

    def test_collapseRequests_collapse_all(self):
        if False:
            print('Hello World!')

        def collapseRequests_fn(master, builder, brdict1, brdict2):
            if False:
                print('Hello World!')
            return True
        self.bldr.getCollapseRequestsFn = lambda : collapseRequests_fn
        return self.do_request_collapse(self.BASE_ROWS, [21], [19, 20])

    def test_collapseRequests_collapse_all_duplicates(self):
        if False:
            i = 10
            return i + 15

        def collapseRequests_fn(master, builder, brdict1, brdict2):
            if False:
                return 10
            return True
        self.bldr.getCollapseRequestsFn = lambda : collapseRequests_fn
        return self.do_request_collapse(self.BASE_ROWS, [21, 21], [19, 20])

    def makeBuildRequestRows(self, brid, bsid, changeid, ssid, codebase, branch=None, project=None, repository=None, patchid=None, revision=None, bs_properties=None):
        if False:
            for i in range(10):
                print('nop')
        rows = [fakedb.SourceStamp(id=ssid, codebase=codebase, branch=branch, project=project, repository=repository, patchid=patchid, revision=revision), fakedb.Buildset(id=bsid, reason='foo', submitted_at=1300305712, results=-1), fakedb.BuildsetSourceStamp(sourcestampid=ssid, buildsetid=bsid), fakedb.BuildRequest(id=brid, buildsetid=bsid, builderid=77, priority=13, submitted_at=1300305712, results=-1)]
        if changeid:
            rows.append(fakedb.Change(changeid=changeid, branch='trunk', revision='9283', repository='svn://...', project='world-domination', sourcestampid=ssid))
        if patchid:
            rows.append(fakedb.Patch(id=patchid, patch_base64='aGVsbG8sIHdvcmxk', patch_author='bar', patch_comment='foo', subdir='/foo', patchlevel=3))
        if bs_properties:
            for (prop_name, prop_value) in bs_properties.items():
                rows.append(fakedb.BuildsetProperty(buildsetid=bsid, property_name=prop_name, property_value=json.dumps(prop_value)))
        return rows

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_codebases(self):
        if False:
            i = 10
            return i + 15
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [19, 20])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_does_not_collapse_older(self):
        if False:
            for i in range(10):
                print('nop')
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [19], [])
        yield self.do_request_collapse(rows, [20], [19])
        yield self.do_request_collapse(rows, [21], [20])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_does_not_collapse_concurrent_claims(self):
        if False:
            i = 10
            return i + 15
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C')
        claimed = []

        @defer.inlineCallbacks
        def collapse_fn(master, builder, brdict1, brdict2):
            if False:
                print('Hello World!')
            if not claimed:
                yield self.master.data.updates.claimBuildRequests([20])
                claimed.append(20)
            res = (yield Builder._defaultCollapseRequestFn(master, builder, brdict1, brdict2))
            return res
        self.bldr.getCollapseRequestsFn = lambda : collapse_fn
        yield self.do_request_collapse(rows, [21], [19])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_does_not_collapse_scheduler_props(self):
        if False:
            while True:
                i = 10
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C', bs_properties={'prop': ('value', 'Scheduler')})
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C', bs_properties={'prop': ('value', 'Other source')})
        rows += self.makeBuildRequestRows(19, 119, None, 219, 'C', bs_properties={'prop': ('value2', 'Scheduler')})
        rows += self.makeBuildRequestRows(18, 118, None, 218, 'C', bs_properties={'prop': ('value', 'Scheduler')})
        rows += self.makeBuildRequestRows(17, 117, None, 217, 'C', bs_properties={'prop': ('value3', 'Other source')})
        rows += self.makeBuildRequestRows(16, 116, None, 216, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [21], [18])
        yield self.do_request_collapse(rows, [20], [16, 17])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_codebases_branches(self):
        if False:
            for i in range(10):
                print('nop')
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A', 'br1')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C', 'br2')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C', 'br2')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C', 'br3')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [19])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_codebases_repository(self):
        if False:
            return 10
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A', None, 'p1')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C', None, 'p2')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C', None, 'p2')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C', None, 'p3')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [19])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_codebases_projects(self):
        if False:
            for i in range(10):
                print('nop')
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A', None, None, 'project1')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C', None, None, 'project2')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C', None, None, 'project2')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C', None, None, 'project3')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [19])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_a_patch(self):
        if False:
            return 10
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C', patchid=123)
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [20])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_changes(self):
        if False:
            while True:
                i = 10
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A')
        rows += self.makeBuildRequestRows(21, 121, 123, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C')
        rows += self.makeBuildRequestRows(20, 120, 124, 220, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [20])

    @defer.inlineCallbacks
    def test_collapseRequests_collapse_default_with_non_matching_revision(self):
        if False:
            return 10
        rows = [fakedb.Builder(id=77, name='A')]
        rows += self.makeBuildRequestRows(22, 122, None, 222, 'A')
        rows += self.makeBuildRequestRows(21, 121, None, 221, 'C')
        rows += self.makeBuildRequestRows(19, 119, None, 210, 'C', revision='abcd1234')
        rows += self.makeBuildRequestRows(20, 120, None, 220, 'C')
        self.bldr.getCollapseRequestsFn = lambda : Builder._defaultCollapseRequestFn
        yield self.do_request_collapse(rows, [22], [])
        yield self.do_request_collapse(rows, [21], [20])

class TestSourceStamp(unittest.TestCase):

    def test_asdict_minimal(self):
        if False:
            while True:
                i = 10
        ssdatadict = {'ssid': '123', 'branch': None, 'revision': None, 'patch': None, 'repository': 'testrepo', 'codebase': 'testcodebase', 'project': 'testproject', 'created_at': datetime.datetime(2019, 4, 1, 23, 38, 33, 154354)}
        ss = buildrequest.TempSourceStamp(ssdatadict)
        self.assertEqual(ss.asDict(), {'branch': None, 'codebase': 'testcodebase', 'patch_author': None, 'patch_body': None, 'patch_comment': None, 'patch_level': None, 'patch_subdir': None, 'project': 'testproject', 'repository': 'testrepo', 'revision': None})

    def test_asdict_no_patch(self):
        if False:
            print('Hello World!')
        ssdatadict = {'ssid': '123', 'branch': 'testbranch', 'revision': 'testrev', 'patch': None, 'repository': 'testrepo', 'codebase': 'testcodebase', 'project': 'testproject', 'created_at': datetime.datetime(2019, 4, 1, 23, 38, 33, 154354)}
        ss = buildrequest.TempSourceStamp(ssdatadict)
        self.assertEqual(ss.asDict(), {'branch': 'testbranch', 'codebase': 'testcodebase', 'patch_author': None, 'patch_body': None, 'patch_comment': None, 'patch_level': None, 'patch_subdir': None, 'project': 'testproject', 'repository': 'testrepo', 'revision': 'testrev'})

    def test_asdict_with_patch(self):
        if False:
            for i in range(10):
                print('nop')
        ssdatadict = {'ssid': '123', 'branch': 'testbranch', 'revision': 'testrev', 'patch': {'patchid': 1234, 'body': b'testbody', 'level': 2, 'author': 'testauthor', 'comment': 'testcomment', 'subdir': 'testsubdir'}, 'repository': 'testrepo', 'codebase': 'testcodebase', 'project': 'testproject', 'created_at': datetime.datetime(2019, 4, 1, 23, 38, 33, 154354)}
        ss = buildrequest.TempSourceStamp(ssdatadict)
        self.assertEqual(ss.asDict(), {'branch': 'testbranch', 'codebase': 'testcodebase', 'patch_author': 'testauthor', 'patch_body': b'testbody', 'patch_comment': 'testcomment', 'patch_level': 2, 'patch_subdir': 'testsubdir', 'project': 'testproject', 'repository': 'testrepo', 'revision': 'testrev'})

class TestBuildRequest(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()

    @defer.inlineCallbacks
    def test_fromBrdict(self):
        if False:
            i = 10
            return i + 15
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=77, name='bldr'), fakedb.SourceStamp(id=234, branch='trunk', revision='9284', repository='svn://...', project='world-domination'), fakedb.Change(changeid=13, branch='trunk', revision='9283', repository='svn://...', project='world-domination', sourcestampid=234), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=234), fakedb.BuildsetProperty(buildsetid=539, property_name='x', property_value='[1, "X"]'), fakedb.BuildsetProperty(buildsetid=539, property_name='y', property_value='[2, "Y"]'), fakedb.BuildRequest(id=288, buildsetid=539, builderid=77, priority=13, submitted_at=1200000000)])
        brdict = (yield master.db.buildrequests.getBuildRequest(288))
        br = (yield buildrequest.BuildRequest.fromBrdict(master, brdict))
        self.assertEqual([ss.ssid for ss in br.sources.values()], [234])
        self.assertEqual(br.reason, 'triggered')
        self.assertEqual(br.properties.getProperty('x'), 1)
        self.assertEqual(br.properties.getProperty('y'), 2)
        self.assertEqual(br.submittedAt, 1200000000)
        self.assertEqual(br.buildername, 'bldr')
        self.assertEqual(br.priority, 13)
        self.assertEqual(br.id, 288)
        self.assertEqual(br.bsid, 539)

    @defer.inlineCallbacks
    def test_fromBrdict_submittedAt_NULL(self):
        if False:
            print('Hello World!')
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=77, name='bldr'), fakedb.SourceStamp(id=234, branch='trunk', revision='9284', repository='svn://...', project='world-domination'), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=234), fakedb.BuildRequest(id=288, buildsetid=539, builderid=77, priority=13, submitted_at=None)])
        brdict = (yield master.db.buildrequests.getBuildRequest(288))
        br = (yield buildrequest.BuildRequest.fromBrdict(master, brdict))
        self.assertEqual(br.submittedAt, None)

    def test_fromBrdict_no_sourcestamps(self):
        if False:
            return 10
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=78, name='not important'), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildRequest(id=288, buildsetid=539, builderid=78, priority=0, submitted_at=None)])
        d = master.db.buildrequests.getBuildRequest(288)
        d.addCallback(lambda brdict: buildrequest.BuildRequest.fromBrdict(master, brdict))
        return self.assertFailure(d, AssertionError)

    @defer.inlineCallbacks
    def test_fromBrdict_multiple_sourcestamps(self):
        if False:
            print('Hello World!')
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=77, name='bldr'), fakedb.SourceStamp(id=234, branch='trunk', revision='9283', repository='svn://a..', codebase='A', project='world-domination'), fakedb.Change(changeid=13, branch='trunk', revision='9283', repository='svn://a..', codebase='A', project='world-domination', sourcestampid=234), fakedb.SourceStamp(id=235, branch='trunk', revision='9284', repository='svn://b..', codebase='B', project='world-domination'), fakedb.Change(changeid=14, branch='trunk', revision='9284', repository='svn://b..', codebase='B', project='world-domination', sourcestampid=235), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=234), fakedb.BuildsetProperty(buildsetid=539, property_name='x', property_value='[1, "X"]'), fakedb.BuildsetProperty(buildsetid=539, property_name='y', property_value='[2, "Y"]'), fakedb.BuildRequest(id=288, buildsetid=539, builderid=77, priority=13, submitted_at=1200000000)])
        brdict = (yield master.db.buildrequests.getBuildRequest(288))
        br = (yield buildrequest.BuildRequest.fromBrdict(master, brdict))
        self.assertEqual(br.reason, 'triggered')
        self.assertEqual(br.properties.getProperty('x'), 1)
        self.assertEqual(br.properties.getProperty('y'), 2)
        self.assertEqual(br.submittedAt, 1200000000)
        self.assertEqual(br.buildername, 'bldr')
        self.assertEqual(br.priority, 13)
        self.assertEqual(br.id, 288)
        self.assertEqual(br.bsid, 539)

    @defer.inlineCallbacks
    def test_mergeSourceStampsWith_common_codebases(self):
        if False:
            for i in range(10):
                print('nop')
        ' This testcase has two buildrequests\n            Request Change Codebase Revision Comment\n            ----------------------------------------------------------------------\n            288     13     A        9283\n            289     15     A        9284\n            288     14     B        9200\n            289     16     B        9201\n            --------------------------------\n            After merged in Build:\n            Source1 has rev 9284 and contains changes 13 and 15 from repository svn://a\n            Source2 has rev 9201 and contains changes 14 and 16 from repository svn://b\n        '
        brs = []
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=77, name='bldr'), fakedb.SourceStamp(id=234, branch='trunk', revision='9283', repository='svn://a..', codebase='A', project='world-domination'), fakedb.Change(changeid=13, branch='trunk', revision='9283', repository='svn://a..', codebase='A', project='world-domination', sourcestampid=234), fakedb.SourceStamp(id=235, branch='trunk', revision='9200', repository='svn://b..', codebase='B', project='world-domination'), fakedb.Change(changeid=14, branch='trunk', revision='9200', repository='svn://b..', codebase='A', project='world-domination', sourcestampid=235), fakedb.SourceStamp(id=236, branch='trunk', revision='9284', repository='svn://a..', codebase='A', project='world-domination'), fakedb.Change(changeid=15, branch='trunk', revision='9284', repository='svn://a..', codebase='A', project='world-domination', sourcestampid=236), fakedb.SourceStamp(id=237, branch='trunk', revision='9201', repository='svn://b..', codebase='B', project='world-domination'), fakedb.Change(changeid=16, branch='trunk', revision='9201', repository='svn://b..', codebase='B', project='world-domination', sourcestampid=237), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=234), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=235), fakedb.BuildRequest(id=288, buildsetid=539, builderid=77), fakedb.Buildset(id=540, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=540, sourcestampid=236), fakedb.BuildsetSourceStamp(buildsetid=540, sourcestampid=237), fakedb.BuildRequest(id=289, buildsetid=540, builderid=77)])
        brdict = (yield master.db.buildrequests.getBuildRequest(288))
        res = (yield buildrequest.BuildRequest.fromBrdict(master, brdict))
        brs.append(res)
        brdict = (yield master.db.buildrequests.getBuildRequest(289))
        res = (yield buildrequest.BuildRequest.fromBrdict(master, brdict))
        brs.append(res)
        sources = brs[0].mergeSourceStampsWith(brs[1:])
        source1 = source2 = None
        for source in sources:
            if source.codebase == 'A':
                source1 = source
            if source.codebase == 'B':
                source2 = source
        self.assertFalse(source1 is None)
        self.assertEqual(source1.revision, '9284')
        self.assertFalse(source2 is None)
        self.assertEqual(source2.revision, '9201')

    @defer.inlineCallbacks
    def test_canBeCollapsed_different_codebases_raises_error(self):
        if False:
            while True:
                i = 10
        ' This testcase has two buildrequests\n            Request Change Codebase   Revision Comment\n            ----------------------------------------------------------------------\n            288     17     C          1800     request 1 has repo not in request 2\n            289     18     D          2100     request 2 has repo not in request 1\n            --------------------------------\n            Merge cannot be performed and raises error:\n              Merging requests requires both requests to have the same codebases\n        '
        brDicts = []
        master = fakemaster.make_master(self, wantData=True, wantDb=True)
        master.db.insert_test_data([fakedb.Builder(id=77, name='bldr'), fakedb.SourceStamp(id=238, branch='trunk', revision='1800', repository='svn://c..', codebase='C', project='world-domination'), fakedb.Change(changeid=17, branch='trunk', revision='1800', repository='svn://c..', codebase='C', project='world-domination', sourcestampid=238), fakedb.SourceStamp(id=239, branch='trunk', revision='2100', repository='svn://d..', codebase='D', project='world-domination'), fakedb.Change(changeid=18, branch='trunk', revision='2100', repository='svn://d..', codebase='D', project='world-domination', sourcestampid=239), fakedb.Buildset(id=539, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=539, sourcestampid=238), fakedb.BuildRequest(id=288, buildsetid=539, builderid=77), fakedb.Buildset(id=540, reason='triggered'), fakedb.BuildsetSourceStamp(buildsetid=540, sourcestampid=239), fakedb.BuildRequest(id=289, buildsetid=540, builderid=77)])
        req = (yield master.db.buildrequests.getBuildRequest(288))
        brDicts.append(req)
        req = (yield master.db.buildrequests.getBuildRequest(289))
        brDicts.append(req)
        can_collapse = (yield buildrequest.BuildRequest.canBeCollapsed(master, brDicts[0], brDicts[1]))
        self.assertEqual(can_collapse, False)