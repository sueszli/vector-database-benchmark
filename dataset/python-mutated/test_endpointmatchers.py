from twisted.internet import defer
from twisted.trial import unittest
from buildbot.schedulers.forcesched import ForceScheduler
from buildbot.test import fakedb
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import www
from buildbot.www.authz import endpointmatchers

class EndpointBase(TestReactorMixin, www.WwwTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.master = self.make_master(url='h:/a/b/')
        self.db = self.master.db
        self.matcher = self.makeMatcher()
        self.matcher.setAuthz(self.master.authz)
        self.insertData()

    def makeMatcher(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def assertMatch(self, match):
        if False:
            i = 10
            return i + 15
        self.assertTrue(match is not None)

    def assertNotMatch(self, match):
        if False:
            i = 10
            return i + 15
        self.assertTrue(match is None)

    def insertData(self):
        if False:
            i = 10
            return i + 15
        self.db.insert_test_data([fakedb.SourceStamp(id=13, branch='secret'), fakedb.Build(id=15, buildrequestid=16, masterid=1, workerid=2, builderid=21), fakedb.BuildRequest(id=16, buildsetid=17), fakedb.Buildset(id=17), fakedb.BuildsetSourceStamp(id=20, buildsetid=17, sourcestampid=13), fakedb.Builder(id=21, name='builder')])

class ValidEndpointMixin:

    @defer.inlineCallbacks
    def test_invalidPath(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('foo', 'bar')))
        self.assertNotMatch(ret)

class AnyEndpointMatcher(EndpointBase):

    def makeMatcher(self):
        if False:
            while True:
                i = 10
        return endpointmatchers.AnyEndpointMatcher(role='foo')

    @defer.inlineCallbacks
    def test_nominal(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('foo', 'bar')))
        self.assertMatch(ret)

class AnyControlEndpointMatcher(EndpointBase):

    def makeMatcher(self):
        if False:
            print('Hello World!')
        return endpointmatchers.AnyControlEndpointMatcher(role='foo')

    @defer.inlineCallbacks
    def test_default_action(self):
        if False:
            while True:
                i = 10
        ret = (yield self.matcher.match(('foo', 'bar')))
        self.assertMatch(ret)

    @defer.inlineCallbacks
    def test_get(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('foo', 'bar'), action='GET'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_other_action(self):
        if False:
            for i in range(10):
                print('nop')
        ret = (yield self.matcher.match(('foo', 'bar'), action='foo'))
        self.assertMatch(ret)

class ViewBuildsEndpointMatcherBranch(EndpointBase, ValidEndpointMixin):

    def makeMatcher(self):
        if False:
            i = 10
            return i + 15
        return endpointmatchers.ViewBuildsEndpointMatcher(branch='secret', role='agent')

    @defer.inlineCallbacks
    def test_build(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('builds', '15')))
        self.assertMatch(ret)
    test_build.skip = 'ViewBuildsEndpointMatcher is not implemented yet'

class StopBuildEndpointMatcherBranch(EndpointBase, ValidEndpointMixin):

    def makeMatcher(self):
        if False:
            i = 10
            return i + 15
        return endpointmatchers.StopBuildEndpointMatcher(builder='builder', role='owner')

    @defer.inlineCallbacks
    def test_build(self):
        if False:
            return 10
        ret = (yield self.matcher.match(('builds', '15'), 'stop'))
        self.assertMatch(ret)

    @defer.inlineCallbacks
    def test_build_no_match(self):
        if False:
            i = 10
            return i + 15
        self.matcher.builder = 'foo'
        ret = (yield self.matcher.match(('builds', '15'), 'stop'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_build_no_builder(self):
        if False:
            print('Hello World!')
        self.matcher.builder = None
        ret = (yield self.matcher.match(('builds', '15'), 'stop'))
        self.assertMatch(ret)

class ForceBuildEndpointMatcherBranch(EndpointBase, ValidEndpointMixin):

    def makeMatcher(self):
        if False:
            i = 10
            return i + 15
        return endpointmatchers.ForceBuildEndpointMatcher(builder='builder', role='owner')

    def insertData(self):
        if False:
            print('Hello World!')
        super().insertData()
        self.master.allSchedulers = lambda : [ForceScheduler(name='sched1', builderNames=['builder'])]

    @defer.inlineCallbacks
    def test_build(self):
        if False:
            print('Hello World!')
        ret = (yield self.matcher.match(('builds', '15'), 'stop'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_forcesched(self):
        if False:
            while True:
                i = 10
        ret = (yield self.matcher.match(('forceschedulers', 'sched1'), 'force'))
        self.assertMatch(ret)

    @defer.inlineCallbacks
    def test_noforcesched(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('forceschedulers', 'sched2'), 'force'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_forcesched_builder_no_match(self):
        if False:
            return 10
        self.matcher.builder = 'foo'
        ret = (yield self.matcher.match(('forceschedulers', 'sched1'), 'force'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_forcesched_nobuilder(self):
        if False:
            print('Hello World!')
        self.matcher.builder = None
        ret = (yield self.matcher.match(('forceschedulers', 'sched1'), 'force'))
        self.assertMatch(ret)

class EnableSchedulerEndpointMatcher(EndpointBase, ValidEndpointMixin):

    def makeMatcher(self):
        if False:
            for i in range(10):
                print('nop')
        return endpointmatchers.EnableSchedulerEndpointMatcher(role='agent')

    @defer.inlineCallbacks
    def test_build(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('builds', '15'), 'stop'))
        self.assertNotMatch(ret)

    @defer.inlineCallbacks
    def test_scheduler_enable(self):
        if False:
            i = 10
            return i + 15
        ret = (yield self.matcher.match(('schedulers', '15'), 'enable'))
        self.assertMatch(ret)