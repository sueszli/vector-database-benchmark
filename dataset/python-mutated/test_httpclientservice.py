import datetime
import json
import os
from unittest import mock
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import components
from twisted.trial import unittest
from twisted.web import resource
from twisted.web import server
from buildbot import interfaces
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.util import bytes2unicode
from buildbot.util import httpclientservice
from buildbot.util import service
from buildbot.util import unicode2bytes
try:
    from requests.auth import HTTPDigestAuth
except ImportError:
    pass
components.registerAdapter(lambda m: m, mock.Mock, interfaces.IHttpResponse)

class HTTPClientServiceTestBase(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        if httpclientservice.txrequests is None or httpclientservice.treq is None:
            raise unittest.SkipTest('this test requires txrequests and treq')
        self.patch(httpclientservice, 'txrequests', mock.Mock())
        self.patch(httpclientservice, 'treq', mock.Mock())
        self.parent = service.MasterService()
        self.parent.reactor = reactor
        self.base_headers = {}
        yield self.parent.startService()

class HTTPClientServiceTestTxRequest(HTTPClientServiceTestBase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        yield super().setUp()
        self._http = (yield httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', headers=self.base_headers))

    @defer.inlineCallbacks
    def test_get(self):
        if False:
            print('Hello World!')
        yield self._http.get('/bar')
        self._http._session.request.assert_called_once_with('get', 'http://foo/bar', headers={}, background_callback=mock.ANY)

    @defer.inlineCallbacks
    def test_get_full_url(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._http.get('http://other/bar')
        self._http._session.request.assert_called_once_with('get', 'http://other/bar', headers={}, background_callback=mock.ANY)

    @defer.inlineCallbacks
    def test_put(self):
        if False:
            return 10
        yield self._http.put('/bar', json={'foo': 'bar'})
        jsonStr = json.dumps({'foo': 'bar'})
        jsonBytes = unicode2bytes(jsonStr)
        headers = {'Content-Type': 'application/json'}
        self._http._session.request.assert_called_once_with('put', 'http://foo/bar', background_callback=mock.ANY, data=jsonBytes, headers=headers)

    @defer.inlineCallbacks
    def test_post(self):
        if False:
            return 10
        yield self._http.post('/bar', json={'foo': 'bar'})
        jsonStr = json.dumps({'foo': 'bar'})
        jsonBytes = unicode2bytes(jsonStr)
        headers = {'Content-Type': 'application/json'}
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data=jsonBytes, headers=headers)

    @defer.inlineCallbacks
    def test_delete(self):
        if False:
            while True:
                i = 10
        yield self._http.delete('/bar')
        self._http._session.request.assert_called_once_with('delete', 'http://foo/bar', background_callback=mock.ANY, headers={})

    @defer.inlineCallbacks
    def test_post_headers(self):
        if False:
            i = 10
            return i + 15
        self.base_headers.update({'X-TOKEN': 'XXXYYY'})
        yield self._http.post('/bar', json={'foo': 'bar'})
        jsonStr = json.dumps({'foo': 'bar'})
        jsonBytes = unicode2bytes(jsonStr)
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data=jsonBytes, headers={'X-TOKEN': 'XXXYYY', 'Content-Type': 'application/json'})

    @defer.inlineCallbacks
    def test_post_auth(self):
        if False:
            for i in range(10):
                print('nop')
        self._http = (yield httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', auth=('user', 'pa$$')))
        yield self._http.post('/bar', json={'foo': 'bar'})
        jsonStr = json.dumps({'foo': 'bar'})
        jsonBytes = unicode2bytes(jsonStr)
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data=jsonBytes, auth=('user', 'pa$$'), headers={'Content-Type': 'application/json'})

class HTTPClientServiceTestTxRequestNoEncoding(HTTPClientServiceTestBase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        yield super().setUp()
        self._http = self.successResultOf(httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', headers=self.base_headers, skipEncoding=True))

    @defer.inlineCallbacks
    def test_post_raw(self):
        if False:
            i = 10
            return i + 15
        yield self._http.post('/bar', json={'foo': 'bar'})
        jsonStr = json.dumps({'foo': 'bar'})
        headers = {'Content-Type': 'application/json'}
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data=jsonStr, headers=headers)

    @defer.inlineCallbacks
    def test_post_rawlist(self):
        if False:
            return 10
        yield self._http.post('/bar', json=[{'foo': 'bar'}])
        jsonStr = json.dumps([{'foo': 'bar'}])
        headers = {'Content-Type': 'application/json'}
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data=jsonStr, headers=headers)

class HTTPClientServiceTestTReq(HTTPClientServiceTestBase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        yield super().setUp()
        self.patch(httpclientservice.HTTPClientService, 'PREFER_TREQ', True)
        self._http = (yield httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', headers=self.base_headers))

    @defer.inlineCallbacks
    def test_get(self):
        if False:
            return 10
        yield self._http.get('/bar')
        httpclientservice.treq.get.assert_called_once_with('http://foo/bar', agent=mock.ANY, headers={})

    @defer.inlineCallbacks
    def test_put(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._http.put('/bar', json={'foo': 'bar'})
        headers = {'Content-Type': ['application/json']}
        httpclientservice.treq.put.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=b'{"foo": "bar"}', headers=headers)

    @defer.inlineCallbacks
    def test_post(self):
        if False:
            return 10
        yield self._http.post('/bar', json={'foo': 'bar'})
        headers = {'Content-Type': ['application/json']}
        httpclientservice.treq.post.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=b'{"foo": "bar"}', headers=headers)

    @defer.inlineCallbacks
    def test_delete(self):
        if False:
            while True:
                i = 10
        yield self._http.delete('/bar')
        httpclientservice.treq.delete.assert_called_once_with('http://foo/bar', agent=mock.ANY, headers={})

    @defer.inlineCallbacks
    def test_post_headers(self):
        if False:
            while True:
                i = 10
        self.base_headers.update({'X-TOKEN': 'XXXYYY'})
        yield self._http.post('/bar', json={'foo': 'bar'})
        headers = {'Content-Type': ['application/json'], 'X-TOKEN': ['XXXYYY']}
        httpclientservice.treq.post.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=b'{"foo": "bar"}', headers=headers)

    @defer.inlineCallbacks
    def test_post_auth(self):
        if False:
            print('Hello World!')
        self._http = (yield httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', auth=('user', 'pa$$')))
        yield self._http.post('/bar', json={'foo': 'bar'})
        headers = {'Content-Type': ['application/json']}
        httpclientservice.treq.post.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=b'{"foo": "bar"}', auth=('user', 'pa$$'), headers=headers)

    @defer.inlineCallbacks
    def test_post_auth_digest(self):
        if False:
            return 10
        auth = HTTPDigestAuth('user', 'pa$$')
        self._http = (yield httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', auth=auth))
        yield self._http.post('/bar', data={'foo': 'bar'})
        self._http._session.request.assert_called_once_with('post', 'http://foo/bar', background_callback=mock.ANY, data={'foo': 'bar'}, auth=auth, headers={})

class HTTPClientServiceTestTReqNoEncoding(HTTPClientServiceTestBase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        yield super().setUp()
        self.patch(httpclientservice.HTTPClientService, 'PREFER_TREQ', True)
        self._http = self.successResultOf(httpclientservice.HTTPClientService.getService(self.parent, 'http://foo', headers=self.base_headers, skipEncoding=True))

    @defer.inlineCallbacks
    def test_post_raw(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._http.post('/bar', json={'foo': 'bar'})
        json_str = json.dumps({'foo': 'bar'})
        headers = {'Content-Type': ['application/json']}
        httpclientservice.treq.post.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=json_str, headers=headers)

    @defer.inlineCallbacks
    def test_post_rawlist(self):
        if False:
            i = 10
            return i + 15
        yield self._http.post('/bar', json=[{'foo': 'bar'}])
        json_str = json.dumps([{'foo': 'bar'}])
        headers = {'Content-Type': ['application/json']}
        httpclientservice.treq.post.assert_called_once_with('http://foo/bar', agent=mock.ANY, data=json_str, headers=headers)

class MyResource(resource.Resource):
    isLeaf = True

    def render_GET(self, request):
        if False:
            return 10

        def decode(x):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(x, bytes):
                return bytes2unicode(x)
            elif isinstance(x, (list, tuple)):
                return [bytes2unicode(y) for y in x]
            elif isinstance(x, dict):
                newArgs = {}
                for (a, b) in x.items():
                    newArgs[decode(a)] = decode(b)
                return newArgs
            return x
        args = decode(request.args)
        content_type = request.getHeader(b'content-type')
        if content_type == b'application/json':
            jsonBytes = request.content.read()
            jsonStr = bytes2unicode(jsonBytes)
            args['json_received'] = json.loads(jsonStr)
        data = json.dumps(args)
        data = unicode2bytes(data)
        request.setHeader(b'content-type', b'application/json')
        request.setHeader(b'content-length', b'%d' % len(data))
        if request.method == b'HEAD':
            return b''
        return data
    render_HEAD = render_GET
    render_POST = render_GET

class HTTPClientServiceTestTxRequestE2E(unittest.TestCase):
    """The e2e tests must be the same for txrequests and treq

    We just force treq in the other TestCase
    """

    def httpFactory(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return httpclientservice.HTTPClientService.getService(parent, f'http://127.0.0.1:{self.port}')

    def expect(self, *arg, **kwargs):
        if False:
            while True:
                i = 10
        pass

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.timeout = 10
        if httpclientservice.txrequests is None or httpclientservice.treq is None:
            raise unittest.SkipTest('this test requires txrequests and treq')
        site = server.Site(MyResource())
        self.listenport = reactor.listenTCP(0, site)
        self.port = self.listenport.getHost().port
        self.parent = parent = service.MasterService()
        self.parent.reactor = reactor
        yield parent.startService()
        self._http = (yield self.httpFactory(parent))

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.listenport.stopListening()
        yield self.parent.stopService()

    @defer.inlineCallbacks
    def test_content(self):
        if False:
            while True:
                i = 10
        self.expect('get', '/', content_json={})
        res = (yield self._http.get('/'))
        content = (yield res.content())
        self.assertEqual(content, b'{}')

    @defer.inlineCallbacks
    def test_content_with_params(self):
        if False:
            return 10
        self.expect('get', '/', params={'a': 'b'}, content_json={'a': ['b']})
        res = (yield self._http.get('/', params={'a': 'b'}))
        content = (yield res.content())
        self.assertEqual(content, b'{"a": ["b"]}')

    @defer.inlineCallbacks
    def test_post_content_with_params(self):
        if False:
            i = 10
            return i + 15
        self.expect('post', '/', params={'a': 'b'}, content_json={'a': ['b']})
        res = (yield self._http.post('/', params={'a': 'b'}))
        content = (yield res.content())
        self.assertEqual(content, b'{"a": ["b"]}')

    @defer.inlineCallbacks
    def test_put_content_with_data(self):
        if False:
            return 10
        self.expect('post', '/', data={'a': 'b'}, content_json={'a': ['b']})
        res = (yield self._http.post('/', data={'a': 'b'}))
        content = (yield res.content())
        self.assertEqual(content, b'{"a": ["b"]}')

    @defer.inlineCallbacks
    def test_put_content_with_json(self):
        if False:
            i = 10
            return i + 15
        exp_content_json = {'json_received': {'a': 'b'}}
        self.expect('post', '/', json={'a': 'b'}, content_json=exp_content_json)
        res = (yield self._http.post('/', json={'a': 'b'}))
        content = (yield res.content())
        content = bytes2unicode(content)
        content = json.loads(content)
        self.assertEqual(content, exp_content_json)

    @defer.inlineCallbacks
    def test_put_content_with_json_datetime(self):
        if False:
            while True:
                i = 10
        exp_content_json = {'json_received': {'a': 'b', 'ts': 12}}
        dt = datetime.datetime.utcfromtimestamp(12)
        self.expect('post', '/', json={'a': 'b', 'ts': dt}, content_json=exp_content_json)
        res = (yield self._http.post('/', json={'a': 'b', 'ts': dt}))
        content = (yield res.content())
        content = bytes2unicode(content)
        content = json.loads(content)
        self.assertEqual(content, exp_content_json)

    @defer.inlineCallbacks
    def test_json(self):
        if False:
            for i in range(10):
                print('nop')
        self.expect('get', '/', content_json={})
        res = (yield self._http.get('/'))
        content = (yield res.json())
        self.assertEqual(content, {})
        self.assertEqual(res.code, 200)
    NUM_PARALLEL = os.environ.get('BBTEST_NUM_PARALLEL', 5)

    @defer.inlineCallbacks
    def test_lots(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(self.NUM_PARALLEL):
            self.expect('get', '/', params={'a': 'b'}, content_json={'a': ['b']})
        for _ in range(self.NUM_PARALLEL):
            res = (yield self._http.get('/', params={'a': 'b'}))
            content = (yield res.content())
            self.assertEqual(content, b'{"a": ["b"]}')

    @defer.inlineCallbacks
    def test_lots_parallel(self):
        if False:
            print('Hello World!')
        for _ in range(self.NUM_PARALLEL):
            self.expect('get', '/', params={'a': 'b'}, content_json={'a': ['b']})

        def oneReq():
            if False:
                return 10
            d = self._http.get('/', params={'a': 'b'})

            @d.addCallback
            def content(res):
                if False:
                    return 10
                return res.content()
            return d
        dl = [oneReq() for i in range(self.NUM_PARALLEL)]
        yield defer.gatherResults(dl)

class HTTPClientServiceTestTReqE2E(HTTPClientServiceTestTxRequestE2E):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.patch(httpclientservice.HTTPClientService, 'PREFER_TREQ', True)
        yield super().setUp()

class HTTPClientServiceTestFakeE2E(HTTPClientServiceTestTxRequestE2E):

    @defer.inlineCallbacks
    def httpFactory(self, parent):
        if False:
            print('Hello World!')
        service = (yield fakehttpclientservice.HTTPClientService.getService(parent, self, f'http://127.0.0.1:{self.port}'))
        return service

    def expect(self, *arg, **kwargs):
        if False:
            while True:
                i = 10
        self._http.expect(*arg, **kwargs)