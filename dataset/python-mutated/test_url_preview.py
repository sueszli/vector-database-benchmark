import base64
import json
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple, Type
from urllib.parse import quote, urlencode
from twisted.internet._resolver import HostResolution
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IAddress, IResolutionReceiver
from twisted.test.proto_helpers import AccumulatingProtocol, MemoryReactor
from twisted.web.resource import Resource
from synapse.config.oembed import OEmbedEndpointConfig
from synapse.media.url_previewer import IMAGE_CACHE_EXPIRY_MS
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from synapse.util.stringutils import parse_and_validate_mxc_uri
from tests import unittest
from tests.server import FakeTransport
from tests.test_utils import SMALL_PNG
try:
    import lxml
except ImportError:
    lxml = None

class URLPreviewTests(unittest.HomeserverTestCase):
    if not lxml:
        skip = 'url preview feature requires lxml'
    hijack_auth = True
    user_id = '@test:user'
    end_content = b'<html><head><meta property="og:title" content="~matrix~" /><meta property="og:description" content="hi" /></head></html>'

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        config = self.default_config()
        config['url_preview_enabled'] = True
        config['max_spider_size'] = 9999999
        config['url_preview_ip_range_blacklist'] = ('192.168.1.1', '1.0.0.0/8', '3fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff', '2001:800::/21')
        config['url_preview_ip_range_whitelist'] = ('1.1.1.1',)
        config['url_preview_accept_language'] = ['en-UK', 'en-US;q=0.9', 'fr;q=0.8', '*;q=0.7']
        self.storage_path = self.mktemp()
        self.media_store_path = self.mktemp()
        os.mkdir(self.storage_path)
        os.mkdir(self.media_store_path)
        config['media_store_path'] = self.media_store_path
        provider_config = {'module': 'synapse.media.storage_provider.FileStorageProviderBackend', 'store_local': True, 'store_synchronous': False, 'store_remote': True, 'config': {'directory': self.storage_path}}
        config['media_storage_providers'] = [provider_config]
        hs = self.setup_test_homeserver(config=config)
        hs.config.oembed.oembed_patterns = [OEmbedEndpointConfig(api_endpoint='http://publish.twitter.com/oembed', url_patterns=[re.compile('http://twitter\\.com/.+/status/.+')], formats=None), OEmbedEndpointConfig(api_endpoint='http://www.hulu.com/api/oembed.{format}', url_patterns=[re.compile('http://www\\.hulu\\.com/watch/.+')], formats=['json'])]
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.media_repo = hs.get_media_repository()
        assert self.media_repo.url_previewer is not None
        self.url_previewer = self.media_repo.url_previewer
        self.lookups: Dict[str, Any] = {}

        class Resolver:

            def resolveHostName(_self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IResolutionReceiver:
                if False:
                    i = 10
                    return i + 15
                resolution = HostResolution(hostName)
                resolutionReceiver.resolutionBegan(resolution)
                if hostName not in self.lookups:
                    raise DNSLookupError('OH NO')
                for i in self.lookups[hostName]:
                    resolutionReceiver.addressResolved(i[0]('TCP', i[1], portNumber))
                resolutionReceiver.resolutionComplete()
                return resolutionReceiver
        self.reactor.nameResolver = Resolver()

    def create_resource_dict(self) -> Dict[str, Resource]:
        if False:
            print('Hello World!')
        'Create a resource tree for the test server\n\n        A resource tree is a mapping from path to twisted.web.resource.\n\n        The default implementation creates a JsonResource and calls each function in\n        `servlets` to register servlets against it.\n        '
        return {'/_matrix/media': self.hs.get_media_repository_resource()}

    def _assert_small_png(self, json_body: JsonDict) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Assert properties from the SMALL_PNG test image.'
        self.assertTrue(json_body['og:image'].startswith('mxc://'))
        self.assertEqual(json_body['og:image:height'], 1)
        self.assertEqual(json_body['og:image:width'], 1)
        self.assertEqual(json_body['og:image:type'], 'image/png')
        self.assertEqual(json_body['matrix:image:size'], 67)

    def test_cache_returns_correct_type(self) -> None:
        if False:
            i = 10
            return i + 15
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html\r\n\r\n' % (len(self.end_content),) + self.end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})
        self.assertIn('http://matrix.org', self.url_previewer._cache)
        self.url_previewer._cache.pop('http://matrix.org')
        self.assertNotIn('http://matrix.org', self.url_previewer._cache)
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})

    def test_non_ascii_preview_httpequiv(self) -> None:
        if False:
            print('Hello World!')
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        end_content = b'<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1251"/><meta property="og:title" content="\xe4\xea\xe0" /><meta property="og:description" content="hi" /></head></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['og:title'], 'дка')

    def test_video_rejected(self) -> None:
        if False:
            i = 10
            return i + 15
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        end_content = b'anything'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: video/mp4\r\n\r\n' % len(end_content) + end_content)
        self.pump()
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': "Requested file's content type not allowed for this operation: video/mp4"})

    def test_audio_rejected(self) -> None:
        if False:
            return 10
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        end_content = b'anything'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: audio/aac\r\n\r\n' % len(end_content) + end_content)
        self.pump()
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': "Requested file's content type not allowed for this operation: audio/aac"})

    def test_non_ascii_preview_content_type(self) -> None:
        if False:
            return 10
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        end_content = b'<html><head><meta property="og:title" content="\xe4\xea\xe0" /><meta property="og:description" content="hi" /></head></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="windows-1251"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['og:title'], 'дка')

    def test_overlong_title(self) -> None:
        if False:
            print('Hello World!')
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        end_content = b'<html><head><title>' + b'x' * 2000 + b'</title><meta property="og:description" content="hi" /></head></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="windows-1251"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        res = channel.json_body
        self.assertCountEqual(['og:description'], res.keys())

    def test_ipaddr(self) -> None:
        if False:
            print('Hello World!')
        '\n        IP addresses can be previewed directly.\n        '
        self.lookups['example.com'] = [(IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html\r\n\r\n' % (len(self.end_content),) + self.end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})

    def test_blocked_ip_specific(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Blocked IP addresses, found via DNS, are not spidered.\n        '
        self.lookups['example.com'] = [(IPv4Address, '192.168.1.1')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(len(self.reactor.tcpClients), 0)
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'DNS resolution failure during URL preview generation'})

    def test_blocked_ip_range(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Blocked IP ranges, IPs found over DNS, are not spidered.\n        '
        self.lookups['example.com'] = [(IPv4Address, '1.1.1.2')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'DNS resolution failure during URL preview generation'})

    def test_blocked_ip_specific_direct(self) -> None:
        if False:
            print('Hello World!')
        '\n        Blocked IP addresses, accessed directly, are not spidered.\n        '
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://192.168.1.1', shorthand=False)
        self.assertEqual(len(self.reactor.tcpClients), 0)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'IP address blocked'})
        self.assertEqual(channel.code, 403)

    def test_blocked_ip_range_direct(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Blocked IP ranges, accessed directly, are not spidered.\n        '
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://1.1.1.2', shorthand=False)
        self.assertEqual(channel.code, 403)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'IP address blocked'})

    def test_blocked_ip_range_whitelisted_ip(self) -> None:
        if False:
            print('Hello World!')
        '\n        Blocked but then subsequently whitelisted IP addresses can be\n        spidered.\n        '
        self.lookups['example.com'] = [(IPv4Address, '1.1.1.1')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html\r\n\r\n' % (len(self.end_content),) + self.end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})

    def test_blocked_ip_with_external_ip(self) -> None:
        if False:
            return 10
        "\n        If a hostname resolves a blocked IP, even if there's a non-blocked one,\n        it will be rejected.\n        "
        self.lookups['example.com'] = [(IPv4Address, '1.1.1.2'), (IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'DNS resolution failure during URL preview generation'})

    def test_blocked_ipv6_specific(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Blocked IP addresses, found via DNS, are not spidered.\n        '
        self.lookups['example.com'] = [(IPv6Address, '3fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(len(self.reactor.tcpClients), 0)
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'DNS resolution failure during URL preview generation'})

    def test_blocked_ipv6_range(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Blocked IP ranges, IPs found over DNS, are not spidered.\n        '
        self.lookups['example.com'] = [(IPv6Address, '2001:800::1')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(channel.code, 502)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'DNS resolution failure during URL preview generation'})

    def test_OPTIONS(self) -> None:
        if False:
            print('Hello World!')
        '\n        OPTIONS returns the OPTIONS.\n        '
        channel = self.make_request('OPTIONS', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False)
        self.assertEqual(channel.code, 204)

    def test_accept_language_config_option(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Accept-Language header is sent to the remote server\n        '
        self.lookups['example.com'] = [(IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://example.com', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html\r\n\r\n' % (len(self.end_content),) + self.end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'og:title': '~matrix~', 'og:description': 'hi'})
        self.assertIn(b'Accept-Language: en-UK\r\nAccept-Language: en-US;q=0.9\r\nAccept-Language: fr;q=0.8\r\nAccept-Language: *;q=0.7', server.data)

    def test_image(self) -> None:
        if False:
            while True:
                i = 10
        'An image should be precached if mentioned in the HTML.'
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['cdn.matrix.org'] = [(IPv4Address, '10.1.2.4')]
        result = b'<html><body><img src="http://cdn.matrix.org/foo.png"></body></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        client = self.reactor.tcpClients[1][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: image/png\r\n\r\n' % (len(SMALL_PNG),) + SMALL_PNG)
        self.pump()
        self.assertEqual(channel.code, 200)
        self._assert_small_png(channel.json_body)

    def test_nonexistent_image(self) -> None:
        if False:
            while True:
                i = 10
        "If the preview image doesn't exist, ensure some data is returned."
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        result = b'<html><body><img src="http://cdn.matrix.org/foo.jpg"></body></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        self.assertEqual(len(self.reactor.tcpClients), 1)
        self.assertEqual(channel.code, 200)
        self.assertNotIn('og:image', channel.json_body)

    @unittest.override_config({'url_preview_url_blacklist': [{'netloc': 'cdn.matrix.org'}]})
    def test_image_blocked(self) -> None:
        if False:
            while True:
                i = 10
        "If the preview image doesn't exist, ensure some data is returned."
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['cdn.matrix.org'] = [(IPv4Address, '10.1.2.4')]
        result = b'<html><body><img src="http://cdn.matrix.org/foo.jpg"></body></html>'
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        self.assertEqual(len(self.reactor.tcpClients), 1)
        self.assertEqual(channel.code, 200)
        self.assertNotIn('og:image', channel.json_body)

    def test_oembed_failure(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'If the autodiscovered oEmbed URL fails, ensure some data is returned.'
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        result = b'\n        <title>oEmbed Autodiscovery Fail</title>\n        <link rel="alternate" type="application/json+oembed"\n            href="http://example.com/oembed?url=http%3A%2F%2Fmatrix.org&format=json"\n            title="matrixdotorg" />\n        '
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['og:title'], 'oEmbed Autodiscovery Fail')

    def test_data_url(self) -> None:
        if False:
            print('Hello World!')
        '\n        Requesting to preview a data URL is not supported.\n        '
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        data = base64.b64encode(SMALL_PNG).decode()
        query_params = urlencode({'url': f'<html><head><img src="data:image/png;base64,{data}" /></head></html>'})
        channel = self.make_request('GET', f'/_matrix/media/v3/preview_url?{query_params}', shorthand=False)
        self.pump()
        self.assertEqual(channel.code, 500)

    def test_inline_data_url(self) -> None:
        if False:
            return 10
        '\n        An inline image (as a data URL) should be parsed properly.\n        '
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        data = base64.b64encode(SMALL_PNG)
        end_content = b'<html><head><img src="data:image/png;base64,%s" /></head></html>' % (data,)
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://matrix.org', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertEqual(channel.code, 200)
        self._assert_small_png(channel.json_body)

    def test_oembed_photo(self) -> None:
        if False:
            return 10
        "Test an oEmbed endpoint which returns a 'photo' type which redirects the preview to a new URL."
        self.lookups['publish.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['cdn.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        result = {'version': '1.0', 'type': 'photo', 'url': 'http://cdn.twitter.com/matrixdotorg'}
        oembed_content = json.dumps(result).encode('utf-8')
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://twitter.com/matrixdotorg/status/12345', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: application/json; charset="utf8"\r\n\r\n' % (len(oembed_content),) + oembed_content)
        self.pump()
        client = self.reactor.tcpClients[1][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: image/png\r\n\r\n' % (len(SMALL_PNG),) + SMALL_PNG)
        self.pump()
        self.assertIn(b'/matrixdotorg', server.data)
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        self.assertEqual(body['og:url'], 'http://twitter.com/matrixdotorg/status/12345')
        self._assert_small_png(body)

    def test_oembed_rich(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Test an oEmbed endpoint which returns HTML content via the 'rich' type."
        self.lookups['publish.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        result = {'version': '1.0', 'type': 'rich', 'author_name': 'Alice', 'html': '<div>Content Preview</div>'}
        end_content = json.dumps(result).encode('utf-8')
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://twitter.com/matrixdotorg/status/12345', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: application/json; charset="utf8"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertIn(b'\r\nHost: publish.twitter.com\r\n', server.data)
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        self.assertEqual(body, {'og:url': 'http://twitter.com/matrixdotorg/status/12345', 'og:title': 'Alice', 'og:description': 'Content Preview'})

    def test_oembed_format(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test an oEmbed endpoint which requires the format in the URL.'
        self.lookups['www.hulu.com'] = [(IPv4Address, '10.1.2.3')]
        result = {'version': '1.0', 'type': 'rich', 'html': '<div>Content Preview</div>'}
        end_content = json.dumps(result).encode('utf-8')
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://www.hulu.com/watch/12345', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: application/json; charset="utf8"\r\n\r\n' % (len(end_content),) + end_content)
        self.pump()
        self.assertIn(b'/api/oembed.json', server.data)
        self.assertIn(b'format=json', server.data)
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        self.assertEqual(body, {'og:url': 'http://www.hulu.com/watch/12345', 'og:description': 'Content Preview'})

    @unittest.override_config({'url_preview_url_blacklist': [{'netloc': 'publish.twitter.com'}]})
    def test_oembed_blocked(self) -> None:
        if False:
            while True:
                i = 10
        'The oEmbed URL should not be downloaded if the oEmbed URL is blocked.'
        self.lookups['twitter.com'] = [(IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://twitter.com/matrixdotorg/status/12345', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 403, channel.result)

    def test_oembed_autodiscovery(self) -> None:
        if False:
            return 10
        '\n        Autodiscovery works by finding the link in the HTML response and then requesting an oEmbed URL.\n        1. Request a preview of a URL which is not known to the oEmbed code.\n        2. It returns HTML including a link to an oEmbed preview.\n        3. The oEmbed preview is requested and returns a URL for an image.\n        4. The image is requested for thumbnailing.\n        '
        self.lookups['www.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['publish.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['cdn.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        result = b'\n        <link rel="alternate" type="application/json+oembed"\n            href="http://publish.twitter.com/oembed?url=http%3A%2F%2Fcdn.twitter.com%2Fmatrixdotorg%2Fstatus%2F12345&format=json"\n            title="matrixdotorg" />\n        '
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://www.twitter.com/matrixdotorg/status/12345', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        result2 = {'version': '1.0', 'type': 'photo', 'url': 'http://cdn.twitter.com/matrixdotorg'}
        oembed_content = json.dumps(result2).encode('utf-8')
        client = self.reactor.tcpClients[1][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: application/json; charset="utf8"\r\n\r\n' % (len(oembed_content),) + oembed_content)
        self.pump()
        self.assertIn(b'/oembed?', server.data)
        client = self.reactor.tcpClients[2][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: image/png\r\n\r\n' % (len(SMALL_PNG),) + SMALL_PNG)
        self.pump()
        self.assertIn(b'/matrixdotorg', server.data)
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        self.assertEqual(body['og:url'], 'http://www.twitter.com/matrixdotorg/status/12345')
        self._assert_small_png(body)

    @unittest.override_config({'url_preview_url_blacklist': [{'netloc': 'publish.twitter.com'}]})
    def test_oembed_autodiscovery_blocked(self) -> None:
        if False:
            print('Hello World!')
        '\n        If the discovered oEmbed URL is blocked, it should be discarded.\n        '
        self.lookups['www.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        self.lookups['publish.twitter.com'] = [(IPv4Address, '10.1.2.4')]
        result = b'\n        <title>Test</title>\n        <link rel="alternate" type="application/json+oembed"\n            href="http://publish.twitter.com/oembed?url=http%3A%2F%2Fcdn.twitter.com%2Fmatrixdotorg%2Fstatus%2F12345&format=json"\n            title="matrixdotorg" />\n        '
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://www.twitter.com/matrixdotorg/status/12345', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html; charset="utf8"\r\n\r\n' % (len(result),) + result)
        self.pump()
        self.assertEqual(len(self.reactor.tcpClients), 1)
        self.assertIn(b'\r\nHost: www.twitter.com\r\n', server.data)
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        self.assertEqual(body['og:title'], 'Test')
        self.assertNotIn('og:image', body)

    def _download_image(self) -> Tuple[str, str]:
        if False:
            i = 10
            return i + 15
        'Downloads an image into the URL cache.\n        Returns:\n            A (host, media_id) tuple representing the MXC URI of the image.\n        '
        self.lookups['cdn.twitter.com'] = [(IPv4Address, '10.1.2.3')]
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=http://cdn.twitter.com/matrixdotorg', shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: image/png\r\n\r\n' % (len(SMALL_PNG),) + SMALL_PNG)
        self.pump()
        self.assertEqual(channel.code, 200)
        body = channel.json_body
        mxc_uri = body['og:image']
        (host, _port, media_id) = parse_and_validate_mxc_uri(mxc_uri)
        self.assertIsNone(_port)
        return (host, media_id)

    def test_storage_providers_exclude_files(self) -> None:
        if False:
            print('Hello World!')
        'Test that files are not stored in or fetched from storage providers.'
        (host, media_id) = self._download_image()
        rel_file_path = self.media_repo.filepaths.url_cache_filepath_rel(media_id)
        media_store_path = os.path.join(self.media_store_path, rel_file_path)
        storage_provider_path = os.path.join(self.storage_path, rel_file_path)
        self.assertTrue(os.path.isfile(media_store_path))
        self.assertFalse(os.path.isfile(storage_provider_path), 'URL cache file was unexpectedly stored in a storage provider')
        channel = self.make_request('GET', f'/_matrix/media/v3/download/{host}/{media_id}', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 200)
        os.makedirs(os.path.dirname(storage_provider_path), exist_ok=True)
        os.rename(media_store_path, storage_provider_path)
        channel = self.make_request('GET', f'/_matrix/media/v3/download/{host}/{media_id}', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 404, 'URL cache file was unexpectedly retrieved from a storage provider')

    def test_storage_providers_exclude_thumbnails(self) -> None:
        if False:
            return 10
        'Test that thumbnails are not stored in or fetched from storage providers.'
        (host, media_id) = self._download_image()
        rel_thumbnail_path = self.media_repo.filepaths.url_cache_thumbnail_directory_rel(media_id)
        media_store_thumbnail_path = os.path.join(self.media_store_path, rel_thumbnail_path)
        storage_provider_thumbnail_path = os.path.join(self.storage_path, rel_thumbnail_path)
        self.assertTrue(os.path.isdir(media_store_thumbnail_path))
        self.assertFalse(os.path.isdir(storage_provider_thumbnail_path), 'URL cache thumbnails were unexpectedly stored in a storage provider')
        channel = self.make_request('GET', f'/_matrix/media/v3/thumbnail/{host}/{media_id}?width=32&height=32&method=scale', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 200)
        rel_file_path = self.media_repo.filepaths.url_cache_filepath_rel(media_id)
        media_store_path = os.path.join(self.media_store_path, rel_file_path)
        os.remove(media_store_path)
        os.makedirs(os.path.dirname(storage_provider_thumbnail_path), exist_ok=True)
        os.rename(media_store_thumbnail_path, storage_provider_thumbnail_path)
        channel = self.make_request('GET', f'/_matrix/media/v3/thumbnail/{host}/{media_id}?width=32&height=32&method=scale', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 404, 'URL cache thumbnail was unexpectedly retrieved from a storage provider')

    def test_cache_expiry(self) -> None:
        if False:
            while True:
                i = 10
        'Test that URL cache files and thumbnails are cleaned up properly on expiry.'
        (_host, media_id) = self._download_image()
        file_path = self.media_repo.filepaths.url_cache_filepath(media_id)
        file_dirs = self.media_repo.filepaths.url_cache_filepath_dirs_to_delete(media_id)
        thumbnail_dir = self.media_repo.filepaths.url_cache_thumbnail_directory(media_id)
        thumbnail_dirs = self.media_repo.filepaths.url_cache_thumbnail_dirs_to_delete(media_id)
        self.assertTrue(os.path.isfile(file_path))
        self.assertTrue(os.path.isdir(thumbnail_dir))
        self.reactor.advance(IMAGE_CACHE_EXPIRY_MS * 1000 + 1)
        self.get_success(self.url_previewer._expire_url_cache_data())
        for path in [file_path] + file_dirs + [thumbnail_dir] + thumbnail_dirs:
            self.assertFalse(os.path.exists(path), f'{os.path.relpath(path, self.media_store_path)} was not deleted')

    @unittest.override_config({'url_preview_url_blacklist': [{'port': '*'}]})
    def test_blocked_port(self) -> None:
        if False:
            return 10
        "Tests that blocking URLs with a port makes previewing such URLs\n        fail with a 403 error and doesn't impact other previews.\n        "
        self.lookups['matrix.org'] = [(IPv4Address, '10.1.2.3')]
        bad_url = quote('http://matrix.org:8888/foo')
        good_url = quote('http://matrix.org/foo')
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=' + bad_url, shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 403, channel.result)
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=' + good_url, shorthand=False, await_result=False)
        self.pump()
        client = self.reactor.tcpClients[0][2].buildProtocol(None)
        server = AccumulatingProtocol()
        server.makeConnection(FakeTransport(client, self.reactor))
        client.makeConnection(FakeTransport(server, self.reactor))
        client.dataReceived(b'HTTP/1.0 200 OK\r\nContent-Length: %d\r\nContent-Type: text/html\r\n\r\n' % (len(self.end_content),) + self.end_content)
        self.pump()
        self.assertEqual(channel.code, 200)

    @unittest.override_config({'url_preview_url_blacklist': [{'netloc': 'example.com'}]})
    def test_blocked_url(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that blocking URLs with a host makes previewing such URLs\n        fail with a 403 error.\n        '
        self.lookups['example.com'] = [(IPv4Address, '10.1.2.3')]
        bad_url = quote('http://example.com/foo')
        channel = self.make_request('GET', '/_matrix/media/v3/preview_url?url=' + bad_url, shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 403, channel.result)