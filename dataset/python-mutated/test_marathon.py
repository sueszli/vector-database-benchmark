from twisted.internet import defer
from twisted.trial import unittest
from buildbot.interfaces import LatentWorkerSubstantiatiationCancelled
from buildbot.process.properties import Properties
from buildbot.test.fake import fakebuild
from buildbot.test.fake import fakemaster
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.test.fake.fakeprotocol import FakeTrivialConnection as FakeBot
from buildbot.test.reactor import TestReactorMixin
from buildbot.worker.marathon import MarathonLatentWorker

class TestMarathonLatentWorker(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.build = Properties(image='busybox:latest', builder='docker_worker')
        self.worker = None

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.worker is not None:

            class FakeResult:
                code = 200
            self._http.delete = lambda _: defer.succeed(FakeResult())
            self.worker.master.stopService()
        self.flushLoggedErrors(LatentWorkerSubstantiatiationCancelled)

    def test_constructor_normal(self):
        if False:
            while True:
                i = 10
        worker = MarathonLatentWorker('bot', 'tcp://marathon.local', 'foo', 'bar', 'debian:wheezy')
        self.assertEqual(worker._http, None)

    @defer.inlineCallbacks
    def makeWorker(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.setdefault('image', 'debian:wheezy')
        worker = MarathonLatentWorker('bot', 'tcp://marathon.local', **kwargs)
        self.worker = worker
        master = fakemaster.make_master(self, wantData=True)
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(master, self, 'tcp://marathon.local', auth=kwargs.get('auth')))
        yield worker.setServiceParent(master)
        worker.reactor = self.reactor
        yield master.startService()
        worker.masterhash = 'masterhash'
        return worker

    @defer.inlineCallbacks
    def test_builds_may_be_incompatible(self):
        if False:
            print('Hello World!')
        worker = self.worker = (yield self.makeWorker())
        self.assertEqual(worker.builds_may_be_incompatible, True)

    @defer.inlineCallbacks
    def test_start_service(self):
        if False:
            print('Hello World!')
        worker = self.worker = (yield self.makeWorker())
        self.assertNotEqual(worker._http, None)

    @defer.inlineCallbacks
    def test_start_worker(self):
        if False:
            return 10
        worker = (yield self.makeWorker())
        worker.password = 'pass'
        worker.masterFQDN = 'master'
        self._http.expect(method='delete', ep='/v2/apps/buildbot-worker/buildbot-bot-masterhash')
        self._http.expect(method='post', ep='/v2/apps', json={'instances': 1, 'container': {'docker': {'image': 'rendered:debian:wheezy', 'network': 'BRIDGE'}, 'type': 'DOCKER'}, 'id': 'buildbot-worker/buildbot-bot-masterhash', 'env': {'BUILDMASTER': 'master', 'BUILDMASTER_PORT': '1234', 'WORKERNAME': 'bot', 'WORKERPASS': 'pass'}}, code=201, content_json={'Id': 'id'})
        d = worker.substantiate(None, fakebuild.FakeBuildForRendering())
        worker.attached(FakeBot())
        yield d
        self.assertEqual(worker.instance, {'Id': 'id'})

    @defer.inlineCallbacks
    def test_start_worker_but_no_connection_and_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        worker = (yield self.makeWorker())
        worker.password = 'pass'
        worker.masterFQDN = 'master'
        self._http.expect(method='delete', ep='/v2/apps/buildbot-worker/buildbot-bot-masterhash')
        self._http.expect(method='post', ep='/v2/apps', json={'instances': 1, 'container': {'docker': {'image': 'rendered:debian:wheezy', 'network': 'BRIDGE'}, 'type': 'DOCKER'}, 'id': 'buildbot-worker/buildbot-bot-masterhash', 'env': {'BUILDMASTER': 'master', 'BUILDMASTER_PORT': '1234', 'WORKERNAME': 'bot', 'WORKERPASS': 'pass'}}, code=201, content_json={'Id': 'id'})
        worker.substantiate(None, fakebuild.FakeBuildForRendering())
        self.assertEqual(worker.instance, {'Id': 'id'})

    @defer.inlineCallbacks
    def test_start_worker_but_error(self):
        if False:
            for i in range(10):
                print('nop')
        worker = (yield self.makeWorker())
        self._http.expect(method='delete', ep='/v2/apps/buildbot-worker/buildbot-bot-masterhash')
        self._http.expect(method='post', ep='/v2/apps', json={'instances': 1, 'container': {'docker': {'image': 'rendered:debian:wheezy', 'network': 'BRIDGE'}, 'type': 'DOCKER'}, 'id': 'buildbot-worker/buildbot-bot-masterhash', 'env': {'BUILDMASTER': 'master', 'BUILDMASTER_PORT': '1234', 'WORKERNAME': 'bot', 'WORKERPASS': 'pass'}}, code=404, content_json={'message': 'image not found'})
        self._http.expect(method='delete', ep='/v2/apps/buildbot-worker/buildbot-bot-masterhash')
        d = worker.substantiate(None, fakebuild.FakeBuildForRendering())
        self.reactor.advance(0.1)
        with self.assertRaises(Exception):
            yield d
        self.assertEqual(worker.instance, None)

    @defer.inlineCallbacks
    def test_start_worker_with_params(self):
        if False:
            i = 10
            return i + 15
        worker = (yield self.makeWorker(marathon_extra_config={'container': {'docker': {'network': None}}, 'env': {'PARAMETER': 'foo'}}))
        worker.password = 'pass'
        worker.masterFQDN = 'master'
        self._http.expect(method='delete', ep='/v2/apps/buildbot-worker/buildbot-bot-masterhash')
        self._http.expect(method='post', ep='/v2/apps', json={'instances': 1, 'container': {'docker': {'image': 'rendered:debian:wheezy', 'network': None}, 'type': 'DOCKER'}, 'id': 'buildbot-worker/buildbot-bot-masterhash', 'env': {'BUILDMASTER': 'master', 'BUILDMASTER_PORT': '1234', 'WORKERNAME': 'bot', 'WORKERPASS': 'pass', 'PARAMETER': 'foo'}}, code=201, content_json={'Id': 'id'})
        d = worker.substantiate(None, fakebuild.FakeBuildForRendering())
        worker.attached(FakeBot())
        yield d
        self.assertEqual(worker.instance, {'Id': 'id'})