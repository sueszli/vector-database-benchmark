from twisted.internet import defer
from twisted.trial import unittest
from buildbot.reporters.zulip import ZulipStatusPush
from buildbot.test.fake import fakemaster
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.config import ConfigErrorsMixin
from buildbot.test.util.logging import LoggingMixin
from buildbot.test.util.reporter import ReporterTestMixin

class TestZulipStatusPush(unittest.TestCase, ReporterTestMixin, LoggingMixin, ConfigErrorsMixin, TestReactorMixin):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.setup_reporter_test()
        self.master = fakemaster.make_master(testcase=self, wantData=True, wantDb=True, wantMq=True)

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.master.running:
            yield self.master.stopService()

    @defer.inlineCallbacks
    def setupZulipStatusPush(self, endpoint='http://example.com', token='123', stream=None):
        if False:
            i = 10
            return i + 15
        self.sp = ZulipStatusPush(endpoint=endpoint, token=token, stream=stream)
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, endpoint, debug=None, verify=None))
        yield self.sp.setServiceParent(self.master)
        yield self.master.startService()

    @defer.inlineCallbacks
    def test_build_started(self):
        if False:
            while True:
                i = 10
        yield self.setupZulipStatusPush(stream='xyz')
        build = (yield self.insert_build_new())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123&stream=xyz', json={'event': 'new', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000001})
        yield self.sp._got_event(('builds', 20, 'new'), build)

    @defer.inlineCallbacks
    def test_build_finished(self):
        if False:
            return 10
        yield self.setupZulipStatusPush(stream='xyz')
        build = (yield self.insert_build_finished())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123&stream=xyz', json={'event': 'finished', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000005, 'results': 0})
        yield self.sp._got_event(('builds', 20, 'finished'), build)

    @defer.inlineCallbacks
    def test_stream_none(self):
        if False:
            i = 10
            return i + 15
        yield self.setupZulipStatusPush(stream=None)
        build = (yield self.insert_build_finished())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123', json={'event': 'finished', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000005, 'results': 0})
        yield self.sp._got_event(('builds', 20, 'finished'), build)

    def test_endpoint_string(self):
        if False:
            print('Hello World!')
        with self.assertRaisesConfigError('Endpoint must be a string'):
            ZulipStatusPush(endpoint=1234, token='abcd')

    def test_token_string(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('Token must be a string'):
            ZulipStatusPush(endpoint='http://example.com', token=1234)

    @defer.inlineCallbacks
    def test_invalid_json_data(self):
        if False:
            while True:
                i = 10
        yield self.setupZulipStatusPush(stream='xyz')
        build = (yield self.insert_build_new())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123&stream=xyz', json={'event': 'new', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000001}, code=500)
        self.setUpLogging()
        yield self.sp._got_event(('builds', 20, 'new'), build)
        self.assertLogged('500: Error pushing build status to Zulip')

    @defer.inlineCallbacks
    def test_invalid_url(self):
        if False:
            print('Hello World!')
        yield self.setupZulipStatusPush(stream='xyz')
        build = (yield self.insert_build_new())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123&stream=xyz', json={'event': 'new', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000001}, code=404)
        self.setUpLogging()
        yield self.sp._got_event(('builds', 20, 'new'), build)
        self.assertLogged('404: Error pushing build status to Zulip')

    @defer.inlineCallbacks
    def test_invalid_token(self):
        if False:
            print('Hello World!')
        yield self.setupZulipStatusPush(stream='xyz')
        build = (yield self.insert_build_new())
        self._http.expect('post', '/api/v1/external/buildbot?api_key=123&stream=xyz', json={'event': 'new', 'buildid': 20, 'buildername': 'Builder0', 'url': 'http://localhost:8080/#/builders/79/builds/0', 'project': 'testProject', 'timestamp': 10000001}, code=401, content_json={'result': 'error', 'msg': 'Invalid API key', 'code': 'INVALID_API_KEY'})
        self.setUpLogging()
        yield self.sp._got_event(('builds', 20, 'new'), build)
        self.assertLogged('401: Error pushing build status to Zulip')