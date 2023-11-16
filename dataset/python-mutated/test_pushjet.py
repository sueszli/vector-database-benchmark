import os
from unittest import SkipTest
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process.properties import Interpolate
from buildbot.process.results import SUCCESS
from buildbot.reporters.pushjet import PushjetNotifier
from buildbot.test.fake import fakemaster
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.config import ConfigErrorsMixin
from buildbot.util import httpclientservice

class TestPushjetNotifier(ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True, wantDb=True, wantMq=True)

    def setupFakeHttp(self, base_url='https://api.pushjet.io'):
        if False:
            i = 10
            return i + 15
        return fakehttpclientservice.HTTPClientService.getService(self.master, self, base_url)

    @defer.inlineCallbacks
    def setupPushjetNotifier(self, secret=Interpolate('1234'), **kwargs):
        if False:
            return 10
        pn = PushjetNotifier(secret, **kwargs)
        yield pn.setServiceParent(self.master)
        yield pn.startService()
        return pn

    @defer.inlineCallbacks
    def test_sendMessage(self):
        if False:
            print('Hello World!')
        _http = (yield self.setupFakeHttp())
        pn = (yield self.setupPushjetNotifier(levels={'passing': 2}))
        _http.expect('post', '/message', data={'secret': '1234', 'level': 2, 'message': 'Test', 'title': 'Tee'}, content_json={'status': 'ok'})
        n = (yield pn.sendMessage([{'body': 'Test', 'subject': 'Tee', 'results': SUCCESS}]))
        j = (yield n.json())
        self.assertEqual(j['status'], 'ok')

    @defer.inlineCallbacks
    def test_sendNotification(self):
        if False:
            for i in range(10):
                print('nop')
        _http = (yield self.setupFakeHttp('https://tests.io'))
        pn = (yield self.setupPushjetNotifier(base_url='https://tests.io'))
        _http.expect('post', '/message', data={'secret': '1234', 'message': 'Test'}, content_json={'status': 'ok'})
        n = (yield pn.sendNotification({'message': 'Test'}))
        j = (yield n.json())
        self.assertEqual(j['status'], 'ok')

    @defer.inlineCallbacks
    def test_sendRealNotification(self):
        if False:
            return 10
        secret = os.environ.get('TEST_PUSHJET_SECRET')
        if secret is None:
            raise SkipTest('real pushjet test runs only if the variable TEST_PUSHJET_SECRET is defined')
        _http = (yield httpclientservice.HTTPClientService.getService(self.master, 'https://api.pushjet.io'))
        yield _http.startService()
        pn = (yield self.setupPushjetNotifier(secret=secret))
        n = (yield pn.sendNotification({'message': 'Buildbot Pushjet test passed!'}))
        j = (yield n.json())
        self.assertEqual(j['status'], 'ok')