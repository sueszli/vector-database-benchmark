import pytest
from mitmproxy.addons import stickycookie
from mitmproxy.test import taddons
from mitmproxy.test import tflow
from mitmproxy.test import tutils as ntutils

def test_domain_match():
    if False:
        for i in range(10):
            print('nop')
    assert stickycookie.domain_match('www.google.com', '.google.com')
    assert stickycookie.domain_match('google.com', '.google.com')

class TestStickyCookie:

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            with pytest.raises(Exception, match='Invalid filter expression'):
                tctx.configure(sc, stickycookie='~b')
            tctx.configure(sc, stickycookie='foo')
            assert sc.flt
            tctx.configure(sc, stickycookie=None)
            assert not sc.flt

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            f = tflow.tflow(resp=True)
            f.response.headers['set-cookie'] = 'foo=bar'
            sc.request(f)
            sc.response(f)
            assert sc.jar
            assert 'cookie' not in f.request.headers
            f = f.copy()
            sc.request(f)
            assert f.request.headers['cookie'] == 'foo=bar'

    def _response(self, sc, cookie, host):
        if False:
            while True:
                i = 10
        f = tflow.tflow(req=ntutils.treq(host=host, port=80), resp=True)
        f.response.headers['Set-Cookie'] = cookie
        sc.response(f)
        return f

    def test_response(self):
        if False:
            while True:
                i = 10
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            c = 'SSID=mooo; domain=.google.com, FOO=bar; Domain=.google.com; Path=/; Expires=Wed, 13-Jan-2021 22:23:01 GMT; Secure; '
            self._response(sc, c, 'host')
            assert not sc.jar.keys()
            self._response(sc, c, 'www.google.com')
            assert sc.jar.keys()
            sc.jar.clear()
            self._response(sc, 'SSID=mooo', 'www.google.com')
            assert list(sc.jar.keys())[0] == ('www.google.com', 80, '/')

    def test_response_multiple(self):
        if False:
            while True:
                i = 10
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            c1 = 'somecookie=test; Path=/'
            c2 = 'othercookie=helloworld; Path=/'
            f = self._response(sc, c1, 'www.google.com')
            f.response.headers['Set-Cookie'] = c2
            sc.response(f)
            googlekey = list(sc.jar.keys())[0]
            assert len(sc.jar[googlekey].keys()) == 2

    def test_response_weird(self):
        if False:
            while True:
                i = 10
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            f = tflow.tflow(req=ntutils.treq(host='www.google.com', port=80), resp=True)
            cs = ['foo/bar=hello', 'foo:bar=world', 'foo@bar=fizz']
            for c in cs:
                f.response.headers['Set-Cookie'] = c
                sc.response(f)
            googlekey = list(sc.jar.keys())[0]
            assert len(sc.jar[googlekey].keys()) == len(cs)

    def test_response_overwrite(self):
        if False:
            print('Hello World!')
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            c1 = 'somecookie=helloworld; Path=/'
            c2 = 'somecookie=newvalue; Path=/'
            f = self._response(sc, c1, 'www.google.com')
            f.response.headers['Set-Cookie'] = c2
            sc.response(f)
            googlekey = list(sc.jar.keys())[0]
            assert len(sc.jar[googlekey]) == 1
            assert sc.jar[googlekey]['somecookie'] == 'newvalue'

    def test_response_delete(self):
        if False:
            while True:
                i = 10
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            f = self._response(sc, 'duffer=zafar; Path=/', 'www.google.com')
            f.response.headers['Set-Cookie'] = 'duffer=; Expires=Thu, 01-Jan-1970 00:00:00 GMT'
            sc.response(f)
            assert not sc.jar.keys()

    def test_request(self):
        if False:
            return 10
        sc = stickycookie.StickyCookie()
        with taddons.context(sc) as tctx:
            tctx.configure(sc, stickycookie='.*')
            f = self._response(sc, 'SSID=mooo', 'www.google.com')
            assert 'cookie' not in f.request.headers
            sc.request(f)
            assert 'cookie' in f.request.headers