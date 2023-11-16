from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.util import httpclientservice
from buildbot.util import service

class myTestedService(service.BuildbotService):
    name = 'myTestedService'

    @defer.inlineCallbacks
    def reconfigService(self, baseurl):
        if False:
            for i in range(10):
                print('nop')
        self._http = (yield httpclientservice.HTTPClientService.getService(self.master, baseurl))

    @defer.inlineCallbacks
    def doGetRoot(self):
        if False:
            for i in range(10):
                print('nop')
        res = (yield self._http.get('/'))
        if res.code != 200:
            raise RuntimeError(f'{res.code}: server did not succeed')
        res_json = (yield res.json())
        return res_json

class Test(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        baseurl = 'http://127.0.0.1:8080'
        self.parent = service.MasterService()
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.parent, self, baseurl))
        self.tested = myTestedService(baseurl)
        yield self.tested.setServiceParent(self.parent)
        yield self.parent.startService()

    @defer.inlineCallbacks
    def test_root(self):
        if False:
            for i in range(10):
                print('nop')
        self._http.expect('get', '/', content_json={'foo': 'bar'})
        response = (yield self.tested.doGetRoot())
        self.assertEqual(response, {'foo': 'bar'})

    @defer.inlineCallbacks
    def test_root_error(self):
        if False:
            i = 10
            return i + 15
        self._http.expect('get', '/', content_json={'foo': 'bar'}, code=404)
        try:
            yield self.tested.doGetRoot()
        except Exception as e:
            self.assertEqual(str(e), '404: server did not succeed')