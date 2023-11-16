import io
from unittest.mock import patch
import pytest
from mitmproxy import flowfilter
from mitmproxy import http
from mitmproxy.test import tflow

class TestParsing:

    def _dump(self, x):
        if False:
            for i in range(10):
                print('nop')
        c = io.StringIO()
        x.dump(fp=c)
        assert c.getvalue()

    def test_parse_err(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='Empty filter'):
            flowfilter.parse('')
        with pytest.raises(ValueError, match='Invalid filter'):
            flowfilter.parse('~b')
        with pytest.raises(ValueError, match='Invalid filter'):
            flowfilter.parse('~h [')

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        assert flowfilter.parse('~q')
        assert flowfilter.parse('~c 10')
        assert flowfilter.parse('~m foobar')
        assert flowfilter.parse('~u foobar')
        assert flowfilter.parse('~q ~c 10')
        assert flowfilter.parse('~replay')
        assert flowfilter.parse('~replayq')
        assert flowfilter.parse('~replays')
        assert flowfilter.parse('~comment .')
        p = flowfilter.parse('~q ~c 10')
        self._dump(p)
        assert len(p.lst) == 2

    def test_non_ascii(self):
        if False:
            i = 10
            return i + 15
        assert flowfilter.parse('~s шгн')

    def test_naked_url(self):
        if False:
            print('Hello World!')
        a = flowfilter.parse('foobar ~h rex')
        assert a.lst[0].expr == 'foobar'
        assert a.lst[1].expr == 'rex'
        self._dump(a)

    def test_quoting(self):
        if False:
            while True:
                i = 10
        a = flowfilter.parse("~u 'foo ~u bar' ~u voing")
        assert a.lst[0].expr == 'foo ~u bar'
        assert a.lst[1].expr == 'voing'
        self._dump(a)
        a = flowfilter.parse('~u foobar')
        assert a.expr == 'foobar'
        a = flowfilter.parse('~u \'foobar\\"\\\'\'')
        assert a.expr == 'foobar"\''
        a = flowfilter.parse('~u "foo \\\'bar"')
        assert a.expr == "foo 'bar"

    def test_nesting(self):
        if False:
            while True:
                i = 10
        a = flowfilter.parse('(~u foobar & ~h voing)')
        assert a.lst[0].expr == 'foobar'
        self._dump(a)

    def test_not(self):
        if False:
            print('Hello World!')
        a = flowfilter.parse('!~h test')
        assert a.itm.expr == 'test'
        a = flowfilter.parse('!(~u test & ~h bar)')
        assert a.itm.lst[0].expr == 'test'
        self._dump(a)

    def test_binaryops(self):
        if False:
            while True:
                i = 10
        a = flowfilter.parse('~u foobar | ~h voing')
        isinstance(a, flowfilter.FOr)
        self._dump(a)
        a = flowfilter.parse('~u foobar & ~h voing')
        isinstance(a, flowfilter.FAnd)
        self._dump(a)

    def test_wideops(self):
        if False:
            while True:
                i = 10
        a = flowfilter.parse("~hq 'header: qvalue'")
        assert isinstance(a, flowfilter.FHeadRequest)
        self._dump(a)

class TestMatchingHTTPFlow:

    def req(self):
        if False:
            while True:
                i = 10
        return tflow.tflow()

    def resp(self):
        if False:
            i = 10
            return i + 15
        return tflow.tflow(resp=True)

    def err(self):
        if False:
            i = 10
            return i + 15
        return tflow.tflow(err=True)

    def q(self, q, o):
        if False:
            print('Hello World!')
        return flowfilter.parse(q)(o)

    def test_http(self):
        if False:
            i = 10
            return i + 15
        s = self.req()
        assert self.q('~http', s)
        assert not self.q('~tcp', s)

    def test_asset(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.resp()
        assert not self.q('~a', s)
        s.response.headers['content-type'] = 'text/javascript'
        assert self.q('~a', s)

    def test_fcontenttype(self):
        if False:
            i = 10
            return i + 15
        q = self.req()
        s = self.resp()
        assert not self.q('~t content', q)
        assert not self.q('~t content', s)
        q.request.headers['content-type'] = 'text/json'
        assert self.q('~t json', q)
        assert self.q('~tq json', q)
        assert not self.q('~ts json', q)
        s.response.headers['content-type'] = 'text/json'
        assert self.q('~t json', s)
        del s.response.headers['content-type']
        s.request.headers['content-type'] = 'text/json'
        assert self.q('~t json', s)
        assert self.q('~tq json', s)
        assert not self.q('~ts json', s)

    def test_freq_fresp(self):
        if False:
            print('Hello World!')
        q = self.req()
        s = self.resp()
        assert self.q('~q', q)
        assert not self.q('~q', s)
        assert not self.q('~s', q)
        assert self.q('~s', s)

    def test_ferr(self):
        if False:
            return 10
        e = self.err()
        assert self.q('~e', e)

    def test_fmarked(self):
        if False:
            print('Hello World!')
        q = self.req()
        assert not self.q('~marked', q)
        q.marked = ':default:'
        assert self.q('~marked', q)

    def test_fmarker_char(self):
        if False:
            i = 10
            return i + 15
        t = tflow.tflow()
        t.marked = ':default:'
        assert not self.q('~marker X', t)
        t.marked = 'X'
        assert self.q('~marker X', t)

    def test_head(self):
        if False:
            while True:
                i = 10
        q = self.req()
        s = self.resp()
        assert not self.q('~h nonexistent', q)
        assert self.q('~h qvalue', q)
        assert self.q('~h header', q)
        assert self.q("~h 'header: qvalue'", q)
        assert self.q("~h 'header: qvalue'", s)
        assert self.q("~h 'header-response: svalue'", s)
        assert self.q("~hq 'header: qvalue'", s)
        assert not self.q("~hq 'header-response: svalue'", s)
        assert self.q("~hq 'header: qvalue'", q)
        assert not self.q("~hq 'header-request: svalue'", q)
        assert not self.q("~hs 'header: qvalue'", s)
        assert self.q("~hs 'header-response: svalue'", s)
        assert not self.q("~hs 'header: qvalue'", q)

    def match_body(self, q, s):
        if False:
            while True:
                i = 10
        assert not self.q('~b nonexistent', q)
        assert self.q('~b content', q)
        assert self.q('~b message', s)
        assert not self.q('~bq nomatch', s)
        assert self.q('~bq content', q)
        assert self.q('~bq content', s)
        assert not self.q('~bq message', q)
        assert not self.q('~bq message', s)
        s.response.text = 'яч'
        assert self.q('~bs яч', s)
        s.response.text = '测试'
        assert self.q('~bs 测试', s)
        s.response.text = 'ॐ'
        assert self.q('~bs ॐ', s)
        s.response.text = 'لله'
        assert self.q('~bs لله', s)
        s.response.text = 'θεός'
        assert self.q('~bs θεός', s)
        s.response.text = 'לוהים'
        assert self.q('~bs לוהים', s)
        s.response.text = '神'
        assert self.q('~bs 神', s)
        s.response.text = '하나님'
        assert self.q('~bs 하나님', s)
        s.response.text = 'Äÿ'
        assert self.q('~bs Äÿ', s)
        assert not self.q('~bs nomatch', s)
        assert not self.q('~bs content', q)
        assert not self.q('~bs content', s)
        assert not self.q('~bs message', q)
        s.response.text = 'message'
        assert self.q('~bs message', s)

    def test_body(self):
        if False:
            return 10
        q = self.req()
        s = self.resp()
        self.match_body(q, s)
        q.request.encode('gzip')
        s.request.encode('gzip')
        s.response.encode('gzip')
        self.match_body(q, s)

    def test_method(self):
        if False:
            for i in range(10):
                print('nop')
        q = self.req()
        assert self.q('~m get', q)
        assert not self.q('~m post', q)
        q.request.method = 'oink'
        assert not self.q('~m get', q)

    def test_domain(self):
        if False:
            return 10
        q = self.req()
        assert self.q('~d address', q)
        assert not self.q('~d none', q)

    def test_url(self):
        if False:
            i = 10
            return i + 15
        q = self.req()
        s = self.resp()
        assert self.q('~u address', q)
        assert self.q('~u address:22/path', q)
        assert not self.q('~u moo/path', q)
        q.request = None
        assert not self.q('~u address', q)
        assert self.q('~u address', s)
        assert self.q('~u address:22/path', s)
        assert not self.q('~u moo/path', s)

    def test_code(self):
        if False:
            return 10
        q = self.req()
        s = self.resp()
        assert not self.q('~c 200', q)
        assert self.q('~c 200', s)
        assert not self.q('~c 201', s)

    def test_src(self):
        if False:
            print('Hello World!')
        q = self.req()
        assert self.q('~src 127.0.0.1', q)
        assert not self.q('~src foobar', q)
        assert self.q('~src :22', q)
        assert not self.q('~src :99', q)
        assert self.q('~src 127.0.0.1:22', q)
        q.client_conn.peername = None
        assert not self.q('~src address:22', q)
        q.client_conn = None
        assert not self.q('~src address:22', q)

    def test_dst(self):
        if False:
            for i in range(10):
                print('nop')
        q = self.req()
        q.server_conn = tflow.tserver_conn()
        assert self.q('~dst address', q)
        assert not self.q('~dst foobar', q)
        assert self.q('~dst :22', q)
        assert not self.q('~dst :99', q)
        assert self.q('~dst address:22', q)
        q.server_conn.address = None
        assert not self.q('~dst address:22', q)
        q.server_conn = None
        assert not self.q('~dst address:22', q)

    def test_and(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.resp()
        assert self.q('~c 200 & ~h head', s)
        assert self.q('~c 200 & ~h head', s)
        assert not self.q('~c 200 & ~h nohead', s)
        assert self.q('(~c 200 & ~h head) & ~b content', s)
        assert not self.q('(~c 200 & ~h head) & ~b nonexistent', s)
        assert not self.q('(~c 200 & ~h nohead) & ~b content', s)

    def test_or(self):
        if False:
            return 10
        s = self.resp()
        assert self.q('~c 200 | ~h nohead', s)
        assert self.q('~c 201 | ~h head', s)
        assert not self.q('~c 201 | ~h nohead', s)
        assert self.q('(~c 201 | ~h nohead) | ~s', s)

    def test_not(self):
        if False:
            while True:
                i = 10
        s = self.resp()
        assert not self.q('! ~c 200', s)
        assert self.q('! ~c 201', s)
        assert self.q('!~c 201 !~c 202', s)
        assert not self.q('!~c 201 !~c 200', s)

    def test_replay(self):
        if False:
            return 10
        f = tflow.tflow()
        assert not self.q('~replay', f)
        f.is_replay = 'request'
        assert self.q('~replay', f)
        assert self.q('~replayq', f)
        assert not self.q('~replays', f)
        f.is_replay = 'response'
        assert self.q('~replay', f)
        assert not self.q('~replayq', f)
        assert self.q('~replays', f)

    def test_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        f = tflow.tflow()
        f.metadata['a'] = 1
        f.metadata['b'] = 'string'
        f.metadata['c'] = {'key': 'value'}
        assert self.q('~meta a', f)
        assert not self.q('~meta no', f)
        assert self.q('~meta string', f)
        assert self.q('~meta key', f)
        assert self.q('~meta value', f)
        assert self.q('~meta "b: string"', f)
        assert self.q('~meta "\'key\': \'value\'"', f)

class TestMatchingDNSFlow:

    def req(self):
        if False:
            print('Hello World!')
        return tflow.tdnsflow()

    def resp(self):
        if False:
            i = 10
            return i + 15
        return tflow.tdnsflow(resp=True)

    def err(self):
        if False:
            i = 10
            return i + 15
        return tflow.tdnsflow(err=True)

    def q(self, q, o):
        if False:
            i = 10
            return i + 15
        return flowfilter.parse(q)(o)

    def test_dns(self):
        if False:
            return 10
        s = self.req()
        assert self.q('~dns', s)
        assert not self.q('~http', s)
        assert not self.q('~tcp', s)

    def test_freq_fresp(self):
        if False:
            i = 10
            return i + 15
        q = self.req()
        s = self.resp()
        assert self.q('~q', q)
        assert not self.q('~q', s)
        assert not self.q('~s', q)
        assert self.q('~s', s)

    def test_ferr(self):
        if False:
            print('Hello World!')
        e = self.err()
        assert self.q('~e', e)

    def test_body(self):
        if False:
            i = 10
            return i + 15
        q = self.req()
        s = self.resp()
        assert not self.q('~b nonexistent', q)
        assert self.q('~b dns.google', q)
        assert self.q('~b 8.8.8.8', s)
        assert not self.q('~bq 8.8.8.8', s)
        assert self.q('~bq dns.google', q)
        assert self.q('~bq dns.google', s)
        assert not self.q('~bs dns.google', q)
        assert self.q('~bs dns.google', s)
        assert self.q('~bs 8.8.8.8', s)

    def test_url(self):
        if False:
            while True:
                i = 10
        f = self.req()
        assert not self.q('~u whatever', f)
        assert self.q('~u dns.google', f)

class TestMatchingTCPFlow:

    def flow(self):
        if False:
            while True:
                i = 10
        return tflow.ttcpflow()

    def err(self):
        if False:
            return 10
        return tflow.ttcpflow(err=True)

    def q(self, q, o):
        if False:
            for i in range(10):
                print('nop')
        return flowfilter.parse(q)(o)

    def test_tcp(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert self.q('~tcp', f)
        assert not self.q('~udp', f)
        assert not self.q('~http', f)
        assert not self.q('~websocket', f)

    def test_ferr(self):
        if False:
            while True:
                i = 10
        e = self.err()
        assert self.q('~e', e)

    def test_body(self):
        if False:
            return 10
        f = self.flow()
        assert self.q('~b hello', f)
        assert self.q('~b me', f)
        assert not self.q('~b nonexistent', f)
        assert self.q('~bq hello', f)
        assert not self.q('~bq me', f)
        assert not self.q('~bq nonexistent', f)
        assert self.q('~bs me', f)
        assert not self.q('~bs hello', f)
        assert not self.q('~bs nonexistent', f)

    def test_src(self):
        if False:
            return 10
        f = self.flow()
        assert self.q('~src 127.0.0.1', f)
        assert not self.q('~src foobar', f)
        assert self.q('~src :22', f)
        assert not self.q('~src :99', f)
        assert self.q('~src 127.0.0.1:22', f)

    def test_dst(self):
        if False:
            print('Hello World!')
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~dst address', f)
        assert not self.q('~dst foobar', f)
        assert self.q('~dst :22', f)
        assert not self.q('~dst :99', f)
        assert self.q('~dst address:22', f)

    def test_and(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello & ~b me', f)
        assert not self.q('~src wrongaddress & ~b hello', f)
        assert self.q('(~src :22 & ~dst :22) & ~b hello', f)
        assert not self.q('(~src address:22 & ~dst :22) & ~b nonexistent', f)
        assert not self.q('(~src address:22 & ~dst :99) & ~b hello', f)

    def test_or(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello | ~b me', f)
        assert self.q('~src :22 | ~b me', f)
        assert not self.q('~src :99 | ~dst :99', f)
        assert self.q('(~src :22 | ~dst :22) | ~b me', f)

    def test_not(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('! ~src :22', f)
        assert self.q('! ~src :99', f)
        assert self.q('!~src :99 !~src :99', f)
        assert not self.q('!~src :99 !~src :22', f)

    def test_request(self):
        if False:
            return 10
        f = self.flow()
        assert not self.q('~q', f)

    def test_response(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert not self.q('~s', f)

    def test_headers(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert not self.q('~h whatever', f)
        assert not self.q('~hq whatever', f)
        assert not self.q('~hs whatever', f)

    def test_content_type(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~t whatever', f)
        assert not self.q('~tq whatever', f)
        assert not self.q('~ts whatever', f)

    def test_code(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~c 200', f)

    def test_domain(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert not self.q('~d whatever', f)

    def test_method(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert not self.q('~m whatever', f)

    def test_url(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~u whatever', f)

class TestMatchingUDPFlow:

    def flow(self):
        if False:
            i = 10
            return i + 15
        return tflow.tudpflow()

    def err(self):
        if False:
            return 10
        return tflow.tudpflow(err=True)

    def q(self, q, o):
        if False:
            for i in range(10):
                print('nop')
        return flowfilter.parse(q)(o)

    def test_udp(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert self.q('~udp', f)
        assert not self.q('~tcp', f)
        assert not self.q('~http', f)
        assert not self.q('~websocket', f)

    def test_ferr(self):
        if False:
            print('Hello World!')
        e = self.err()
        assert self.q('~e', e)

    def test_body(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert self.q('~b hello', f)
        assert self.q('~b me', f)
        assert not self.q('~b nonexistent', f)
        assert self.q('~bq hello', f)
        assert not self.q('~bq me', f)
        assert not self.q('~bq nonexistent', f)
        assert self.q('~bs me', f)
        assert not self.q('~bs hello', f)
        assert not self.q('~bs nonexistent', f)

    def test_src(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert self.q('~src 127.0.0.1', f)
        assert not self.q('~src foobar', f)
        assert self.q('~src :22', f)
        assert not self.q('~src :99', f)
        assert self.q('~src 127.0.0.1:22', f)

    def test_dst(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~dst address', f)
        assert not self.q('~dst foobar', f)
        assert self.q('~dst :22', f)
        assert not self.q('~dst :99', f)
        assert self.q('~dst address:22', f)

    def test_and(self):
        if False:
            print('Hello World!')
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello & ~b me', f)
        assert not self.q('~src wrongaddress & ~b hello', f)
        assert self.q('(~src :22 & ~dst :22) & ~b hello', f)
        assert not self.q('(~src address:22 & ~dst :22) & ~b nonexistent', f)
        assert not self.q('(~src address:22 & ~dst :99) & ~b hello', f)

    def test_or(self):
        if False:
            return 10
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello | ~b me', f)
        assert self.q('~src :22 | ~b me', f)
        assert not self.q('~src :99 | ~dst :99', f)
        assert self.q('(~src :22 | ~dst :22) | ~b me', f)

    def test_not(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert not self.q('! ~src :22', f)
        assert self.q('! ~src :99', f)
        assert self.q('!~src :99 !~src :99', f)
        assert not self.q('!~src :99 !~src :22', f)

    def test_request(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert not self.q('~q', f)

    def test_response(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~s', f)

    def test_headers(self):
        if False:
            print('Hello World!')
        f = self.flow()
        assert not self.q('~h whatever', f)
        assert not self.q('~hq whatever', f)
        assert not self.q('~hs whatever', f)

    def test_content_type(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~t whatever', f)
        assert not self.q('~tq whatever', f)
        assert not self.q('~ts whatever', f)

    def test_code(self):
        if False:
            while True:
                i = 10
        f = self.flow()
        assert not self.q('~c 200', f)

    def test_domain(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert not self.q('~d whatever', f)

    def test_method(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        assert not self.q('~m whatever', f)

    def test_url(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.flow()
        assert not self.q('~u whatever', f)

class TestMatchingWebSocketFlow:

    def flow(self) -> http.HTTPFlow:
        if False:
            for i in range(10):
                print('nop')
        return tflow.twebsocketflow()

    def q(self, q, o):
        if False:
            print('Hello World!')
        return flowfilter.parse(q)(o)

    def test_websocket(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.flow()
        assert self.q('~websocket', f)
        assert not self.q('~tcp', f)
        assert self.q('~http', f)

    def test_handshake(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.flow()
        assert self.q('~websocket', f)
        assert not self.q('~tcp', f)
        assert self.q('~http', f)
        f = tflow.tflow()
        assert not self.q('~websocket', f)
        f = tflow.tflow(resp=True)
        assert not self.q('~websocket', f)

    def test_domain(self):
        if False:
            i = 10
            return i + 15
        q = self.flow()
        assert self.q('~d example.com', q)
        assert not self.q('~d none', q)

    def test_url(self):
        if False:
            while True:
                i = 10
        q = self.flow()
        assert self.q('~u example.com', q)
        assert self.q('~u example.com/ws', q)
        assert not self.q('~u moo/path', q)

    def test_body(self):
        if False:
            return 10
        f = self.flow()
        assert self.q('~b hello', f)
        assert self.q('~b me', f)
        assert not self.q('~b nonexistent', f)
        assert self.q('~bq hello', f)
        assert not self.q('~bq me', f)
        assert not self.q('~bq nonexistent', f)
        assert self.q('~bs me', f)
        assert not self.q('~bs hello', f)
        assert not self.q('~bs nonexistent', f)

    def test_src(self):
        if False:
            return 10
        f = self.flow()
        assert self.q('~src 127.0.0.1', f)
        assert not self.q('~src foobar', f)
        assert self.q('~src :22', f)
        assert not self.q('~src :99', f)
        assert self.q('~src 127.0.0.1:22', f)

    def test_dst(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~dst address', f)
        assert not self.q('~dst foobar', f)
        assert self.q('~dst :22', f)
        assert not self.q('~dst :99', f)
        assert self.q('~dst address:22', f)

    def test_and(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello & ~b me', f)
        assert not self.q('~src wrongaddress & ~b hello', f)
        assert self.q('(~src :22 & ~dst :22) & ~b hello', f)
        assert not self.q('(~src address:22 & ~dst :22) & ~b nonexistent', f)
        assert not self.q('(~src address:22 & ~dst :99) & ~b hello', f)

    def test_or(self):
        if False:
            i = 10
            return i + 15
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~b hello | ~b me', f)
        assert self.q('~src :22 | ~b me', f)
        assert not self.q('~src :99 | ~dst :99', f)
        assert self.q('(~src :22 | ~dst :22) | ~b me', f)

    def test_not(self):
        if False:
            return 10
        f = self.flow()
        assert not self.q('! ~src :22', f)
        assert self.q('! ~src :99', f)
        assert self.q('!~src :99 !~src :99', f)
        assert not self.q('!~src :99 !~src :22', f)

class TestMatchingDummyFlow:

    def flow(self):
        if False:
            i = 10
            return i + 15
        return tflow.tdummyflow()

    def err(self):
        if False:
            return 10
        return tflow.tdummyflow(err=True)

    def q(self, q, o):
        if False:
            print('Hello World!')
        return flowfilter.parse(q)(o)

    def test_filters(self):
        if False:
            for i in range(10):
                print('nop')
        e = self.err()
        f = self.flow()
        f.server_conn = tflow.tserver_conn()
        assert self.q('~all', f)
        assert not self.q('~a', f)
        assert not self.q('~b whatever', f)
        assert not self.q('~bq whatever', f)
        assert not self.q('~bs whatever', f)
        assert not self.q('~c 0', f)
        assert not self.q('~d whatever', f)
        assert self.q('~dst address', f)
        assert not self.q('~dst nonexistent', f)
        assert self.q('~e', e)
        assert not self.q('~e', f)
        assert not self.q('~http', f)
        assert not self.q('~tcp', f)
        assert not self.q('~websocket', f)
        assert not self.q('~h whatever', f)
        assert not self.q('~hq whatever', f)
        assert not self.q('~hs whatever', f)
        assert not self.q('~m whatever', f)
        assert not self.q('~s', f)
        assert self.q('~src 127.0.0.1', f)
        assert not self.q('~src nonexistent', f)
        assert not self.q('~tcp', f)
        assert not self.q('~t whatever', f)
        assert not self.q('~tq whatever', f)
        assert not self.q('~ts whatever', f)
        assert not self.q('~u whatever', f)
        assert not self.q('~q', f)
        assert not self.q('~comment .', f)
        f.comment = 'comment'
        assert self.q('~comment .', f)

@patch('traceback.extract_tb')
def test_pyparsing_bug(extract_tb):
    if False:
        while True:
            i = 10
    'https://github.com/mitmproxy/mitmproxy/issues/1087'
    extract_tb.return_value = [('', 1, 'test', None)]
    assert flowfilter.parse('test')

def test_match():
    if False:
        return 10
    with pytest.raises(ValueError):
        flowfilter.match('[foobar', None)
    assert flowfilter.match(None, None)
    assert not flowfilter.match('foobar', None)