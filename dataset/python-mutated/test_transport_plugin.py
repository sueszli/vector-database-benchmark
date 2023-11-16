from io import BytesIO
from requests.adapters import BaseAdapter
from requests.models import Response
from requests.utils import get_encoding_from_headers
from httpie.plugins import TransportPlugin
from httpie.plugins.registry import plugin_manager
from .utils import HTTP_OK, http
SCHEME = 'http+fake'

class FakeAdapter(BaseAdapter):

    def send(self, request, **kwargs):
        if False:
            print('Hello World!')
        response = Response()
        response.status_code = 200
        response.reason = 'OK'
        response.headers = {'Content-Type': 'text/html; charset=UTF-8'}
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = BytesIO(b'<!doctype html><html>Hello</html>')
        return response

class FakeTransportPlugin(TransportPlugin):
    name = 'Fake Transport'
    prefix = SCHEME

    def get_adapter(self):
        if False:
            for i in range(10):
                print('nop')
        return FakeAdapter()

def test_transport_from_requests_response(httpbin):
    if False:
        i = 10
        return i + 15
    plugin_manager.register(FakeTransportPlugin)
    try:
        r = http(f'{SCHEME}://example.com')
        assert HTTP_OK in r
        assert 'Hello' in r
        assert 'Content-Type: text/html; charset=UTF-8' in r
    finally:
        plugin_manager.unregister(FakeTransportPlugin)