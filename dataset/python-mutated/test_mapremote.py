import pytest
from mitmproxy.addons import mapremote
from mitmproxy.test import taddons
from mitmproxy.test import tflow

class TestMapRemote:

    def test_configure(self):
        if False:
            while True:
                i = 10
        mr = mapremote.MapRemote()
        with taddons.context(mr) as tctx:
            tctx.configure(mr, map_remote=['one/two/three'])
            with pytest.raises(Exception, match='Invalid regular expression'):
                tctx.configure(mr, map_remote=['/foo/+/three'])

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        mr = mapremote.MapRemote()
        with taddons.context(mr) as tctx:
            tctx.configure(mr, map_remote=[':example.org/images/:mitmproxy.org/img/'])
            f = tflow.tflow()
            f.request.url = b'https://example.org/images/test.jpg'
            mr.request(f)
            assert f.request.url == 'https://mitmproxy.org/img/test.jpg'

    def test_host_header(self):
        if False:
            for i in range(10):
                print('nop')
        mr = mapremote.MapRemote()
        with taddons.context(mr) as tctx:
            tctx.configure(mr, map_remote=['|http://[^/]+|http://example.com:4444'])
            f = tflow.tflow()
            f.request.url = b'http://example.org/example'
            f.request.headers['Host'] = 'example.org'
            mr.request(f)
            assert f.request.headers.get('Host', '') == 'example.com:4444'

    def test_is_killed(self):
        if False:
            i = 10
            return i + 15
        mr = mapremote.MapRemote()
        with taddons.context(mr) as tctx:
            tctx.configure(mr, map_remote=[':example.org:mitmproxy.org'])
            f = tflow.tflow()
            f.request.url = b'https://example.org/images/test.jpg'
            f.kill()
            mr.request(f)
            assert f.request.url == 'https://example.org/images/test.jpg'